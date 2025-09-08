from functools import partial
import torch
from torch import nn
import numpy as np
from scipy import ndimage as ndi
from collections import defaultdict
from tqdm import tqdm
from itertools import chain
from funcmol.models.egnn import EGNNVectorField

class Decoder(nn.Module):
    def __init__(self, config, device):
        """
        Initializes the Decoder class.

        Args:
            config (dict): Configuration dictionary containing parameters for the decoder.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.device = device
        self.grid_dim = config["grid_size"]  # Add grid_dim attribute
        self.n_channels = config["n_channels"]  # Add n_channels attribute
        
        # Initialize coords using get_grid function
        _, self.coords = get_grid(self.grid_dim, resolution=0.25)
        self.coords = self.coords.to(device)
        
        self.net = EGNNVectorField(
            grid_size=config["grid_size"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["n_layers"],
            radius=config["radius"],
            n_atom_types=config["n_channels"],
            code_dim=config["code_dim"],
            cutoff=config.get("cutoff", None),  # Add cutoff parameter with default None
            anchor_spacing=config.get("anchor_spacing", 2.0),  # Add anchor_spacing parameter with default 2.0
            device=device,
            k_neighbors=config.get("k_neighbors", 32)  # Add k_neighbors parameter with default 32
        )

    def forward(self, x, codes):
        """
        x: [B, n_points, 3]
        codes: [B, n_grid, code_dim]
        """
        # Pass the 3D tensors directly to the EGNN, which handles batching internally.
        vector_field = self.net(x, codes)
        
        # The output from EGNN is already in the correct shape: [B, n_points, n_atom_types, 3]
        return vector_field

    def render_code(
        self,
        codes: torch.Tensor,
        batch_size_render: int,
        fabric: object = None,
    ) -> torch.Tensor:
        """
        Renders molecules from given codes.

        Args:
            codes (torch.Tensor): Tensor containing the codes to be rendered.
                                  The codes need to be unnormalized before rendering.
            batch_size_render (int): The size of the batches for rendering.
            fabric (object, optional): An object that provides a print method for logging.
                                       Defaults to None.

        Returns:
            torch.Tensor: A tensor representing the rendered molecules in a grid format.
        """
        # 添加调试信息
        fabric.print(f">> render_code - input codes shape: {codes.shape}")
        fabric.print(f">> render_code - batch_size_render: {batch_size_render}")
        
        # 检查输入
        if torch.isnan(codes).any():
            fabric.print(f">> ERROR: Input codes contains NaN values!")
            raise ValueError("Input codes contains NaN values")
        
        if torch.isinf(codes).any():
            fabric.print(f">> ERROR: Input codes contains Inf values!")
            raise ValueError("Input codes contains Inf values")
        
        # PS: code need to be unnormealized before rendering
        with torch.no_grad():
            if fabric:
                fabric.print(f">> Rendering molecules - batches of {batch_size_render}")
            if codes.device != self.device:
                codes = codes.to(self.device)
            
            # 添加调试信息：检查coords的维度
            if fabric:
                fabric.print(f">> self.coords shape: {self.coords.shape}")
                fabric.print(f">> self.grid_dim: {self.grid_dim}")
                fabric.print(f">> Expected coords size: {self.grid_dim**3}")
            
            # 检查coords维度是否正确
            expected_coords_size = self.grid_dim ** 3
            if self.coords.numel() != expected_coords_size * 3:
                fabric.print(f">> ERROR: coords size mismatch! Expected {expected_coords_size * 3}, got {self.coords.numel()}")
                raise ValueError(f"coords size mismatch: expected {expected_coords_size * 3}, got {self.coords.numel()}")
            
            # 检查并调整batch_size_render，确保不会超过xs的大小
            if batch_size_render > expected_coords_size:
                if fabric:
                    fabric.print(f">> WARNING: batch_size_render ({batch_size_render}) > expected_coords_size ({expected_coords_size})")
                    fabric.print(f">> Adjusting batch_size_render to {expected_coords_size}")
                batch_size_render = expected_coords_size
            
            xs = self.coords.reshape(1, -1, 3)
            if fabric:
                fabric.print(f">> xs shape after reshape: {xs.shape}")
                fabric.print(f">> Final batch_size_render: {batch_size_render}")
            
            # 检查codes的batch大小，如果超过batch_size_render则分批处理
            codes_batch_size = codes.size(0)
            if codes_batch_size > batch_size_render:
                # 分批处理codes
                codes_batches = torch.split(codes, batch_size_render, dim=0)
                if fabric:
                    fabric.print(f">> Splitting codes into {len(codes_batches)} batches for rendering")
                
                pred_list = []
                for i, codes_batch in enumerate(codes_batches):
                    if fabric:
                        fabric.print(f">> Processing codes batch {i+1}/{len(codes_batches)}, codes shape: {codes_batch.shape}")
                    
                    # 为每个codes batch创建对应的xs
                    xs_batch = xs.expand(codes_batch.size(0), -1, -1)  # [batch_size, 729, 3]
                    if fabric:
                        fabric.print(f">> xs_batch shape: {xs_batch.shape}")
                    
                    pred_batch = self.forward_batched(xs_batch, codes_batch, batch_size_render=batch_size_render, threshold=0.2, fabric=fabric)
                    pred_list.append(pred_batch)
                
                pred = torch.cat(pred_list, dim=0)  # 在batch维度上拼接
            else:
                # 直接处理，确保xs和codes的batch维度匹配
                xs_expanded = xs.expand(codes.size(0), -1, -1)  # [batch_size, 729, 3]
                if fabric:
                    fabric.print(f">> xs_expanded shape: {xs_expanded.shape}")
                pred = self.forward_batched(xs_expanded, codes, batch_size_render=batch_size_render, threshold=0.2, fabric=fabric)
            
            # pred的形状是[B, N, T, 3]，需要转换为[B, T, N, 3]然后reshape
            if pred.dim() == 4:  # [B, N, T, 3]
                pred = pred.permute(0, 2, 1, 3)  # [B, T, N, 3]
                grid = pred.reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
            else:  # [B, N, T, 3] -> [B, T, N, 3]
                grid = pred.permute(0, 2, 1).reshape(-1, self.n_channels, self.grid_dim, self.grid_dim, self.grid_dim)
        return grid

    def forward_batched(self, xs, codes, batch_size_render=100_000, threshold=None, to_cpu=True, fabric=None):
        """
        When memory is limited, render the grid in batches.
        """
        # 添加调试信息
        if fabric:
            fabric.print(f">> forward_batched - xs shape: {xs.shape}")
            fabric.print(f">> forward_batched - codes shape: {codes.shape}")
            fabric.print(f">> forward_batched - batch_size_render: {batch_size_render}")
        
        # 现在xs和codes的batch维度应该匹配，直接调用decoder
        try:
            pred = self(xs, codes)
            if to_cpu:
                pred = pred.cpu()
            if threshold is not None:
                pred[pred < threshold] = 0
        except Exception as e:
            if fabric:
                fabric.print(f">> ERROR in forward_batched: {e}")
                fabric.print(f">>   xs shape: {xs.shape}")
                fabric.print(f">>   codes shape: {codes.shape}")
            raise e
        
        return pred

    def codes_to_molecules(
        self,
        codes: torch.Tensor,
        unnormalize: bool = True,
        config: dict = None,
        fabric: object = None,
    ) -> list:
        """
        Convert codes to molecular structures.

        Args:
            codes (torch.Tensor): The input codes representing molecular structures.
            unnormalize (bool): Flag indicating whether to unnormalize the codes.
            config (dict): Configuration dictionary containing parameters for the conversion.
            fabric (optional): An optional fabric object for logging and printing.

        Returns:
            list: A list of refined molecular structures.
        """
        mols_dict = defaultdict(list)
        codes_dict = defaultdict(list)

        # 添加输入验证和调试信息
        fabric.print(f">> codes_to_molecules input - codes shape: {codes.shape}")
        fabric.print(f">> codes_to_molecules input - codes device: {codes.device}")
        
        # 检查输入codes
        if torch.isnan(codes).any():
            fabric.print(f">> ERROR: Input codes contains NaN values!")
            fabric.print(f">> NaN positions: {torch.where(torch.isnan(codes))}")
            raise ValueError("Input codes contains NaN values")
        
        if torch.isinf(codes).any():
            fabric.print(f">> ERROR: Input codes contains Inf values!")
            fabric.print(f">> Inf positions: {torch.where(torch.isinf(codes))}")
            raise ValueError("Input codes contains Inf values")
        
        codes = codes.detach()
        # No longer unnormalize codes
        # if unnormalize:
        #     codes = self.unnormalize_code(codes).detach()

        mols_dict, codes_dict = self.codes_to_grid(
            codes=codes,
            mols_dict=mols_dict,
            codes_dict=codes_dict,
            config=config,
            fabric=fabric,
        )
        fabric.print("(n_atoms, n_samples): ", end=" ")
        for key, value in sorted(mols_dict.items()):
            fabric.print(f"({key}, {len(value)})", end=" ")
        fabric.print()

        # Refine coordinates
        mols = self._refine_coords(
            grouped_mol_inits=mols_dict,
            grouped_codes=codes_dict,
            maxiter=200,
            grid_dim=self.grid_dim,
            resolution=config["dset"]["resolution"],
            fabric=fabric,
        )

        return mols

    def codes_to_grid(
        self,
        codes: torch.Tensor,
        mols_dict: dict = defaultdict(list),
        codes_dict: dict = defaultdict(list),
        config=None,
        fabric=None,
    ) -> tuple:
        """
        Converts a batch of codes to grids and extracts atom coordinates.

        Args:
            batched_codes (torch.Tensor): The batch of codes to convert to grids.
            mols_dict (dict, optional): A dictionary to store molecule objects. Defaults to defaultdict(list).
            codes_dict (dict, optional): A dictionary to store the corresponding codes. Defaults to defaultdict(list).

        Returns:
            tuple: A tuple containing the grid shape, molecule objects, dictionaries of molecule objects, codes, and indices.
        """
        fabric.print(f">> codes_to_grid - input codes shape: {codes.shape}")
        
        # 1. render grid
        fabric.print(f">> Rendering grid with batch_size_render: {config['wjs']['batch_size_render']}")
        grids = self.render_code(codes, config["wjs"]["batch_size_render"], fabric)
        fabric.print(f">> Rendered grids: {len(grids)} grids")
        
        # 检查grids的内容
        for i, grid in enumerate(grids):
            if grid is not None:
                fabric.print(f">> Grid {i} shape: {grid.shape}")
                if torch.isnan(grid).any():
                    fabric.print(f">> WARNING: Grid {i} contains NaN values!")
                if torch.isinf(grid).any():
                    fabric.print(f">> WARNING: Grid {i} contains Inf values!")
            else:
                fabric.print(f">> Grid {i} is None")

        # 2. find peaks (atom coordinates)
        fabric.print(">> Finding peaks")
        for idx, grid in tqdm(enumerate(grids)):
            if grid is None:
                fabric.print(f">> Skipping None grid at index {idx}")
                continue
                
            mol_init = get_atom_coords(grid, rad=config["dset"]["atomic_radius"])
            if mol_init is not None:
                # No longer normalize coordinates
                # mol_init = _normalize_coords(mol_init, self.grid_dim)
                num_coords = int(mol_init["coords"].size(1))
                fabric.print(f">> Grid {idx}: found {num_coords} atoms")
                if num_coords <= 500:
                    mols_dict[num_coords].append(mol_init)
                    codes_dict[num_coords].append(codes[idx].cpu())
                else:
                    fabric.print(f"Molecule {idx} has more than 500 atoms")
            else:
                fabric.print(f"No atoms found in grid {idx}")
        
        # 添加调试信息
        fabric.print(f">> After codes_to_grid - mols_dict keys: {list(mols_dict.keys())}")
        for key, value in mols_dict.items():
            fabric.print(f">>   Key {key}: {len(value)} molecules")
        
        return mols_dict, codes_dict

    def _refine_coords(
        self,
        grouped_mol_inits: dict,
        grouped_codes: dict,
        maxiter: int = 10,
        grid_dim: int = 32,
        resolution: float = 0.25,
        fabric=None,
    ) -> list:
        """
        Refines the coordinates of molecules in batches and handles errors during the process.

        Args:
            grouped_mol_inits (dict): A dictionary where keys are group identifiers and values are lists of initial molecule data.
            grouped_codes (dict): A dictionary where keys are group identifiers and values are lists of codes corresponding to the molecules.
            maxiter (int, optional): Maximum number of iterations for the refinement process. Default is 10.
            grid_dim (int, optional): Dimension of the grid used for normalization. Default is 32.
            resolution (int, optional): Resolution used for normalization. Default is 0.25.
            fabric (optional): An object with a print method for logging messages.

        Returns:
            list: A list of refined molecule data.
        """
        fabric.print(">> Refining molecules")
        for key, mols in tqdm(grouped_mol_inits.items()):
            try:
                coords = self._refine_coords_batch(
                    grouped_mol_inits[key],
                    grouped_codes[key],
                    maxiter=maxiter,
                    fabric=fabric,
                )
                for i in range(coords.size(0)):
                    mols[i]["coords"] = coords[i].unsqueeze(0)
                    # No longer unnormalize coordinates
                    # mols[i] = _unnormalize_coords(mols[i], grid_dim, resolution)
            except Exception as e:
                fabric.print(f"Error refinement: {e}")
                for i in range(len(grouped_mol_inits[key])):
                    try:
                        # No longer unnormalize coordinates
                        # mols[i] = _unnormalize_coords(mols[i], grid_dim, resolution)
                        pass
                    except Exception:
                        fabric.print(
                            f"Error unnormalization: {e} for {i}/{len(grouped_mol_inits[key])}"
                        )
        return list(chain.from_iterable(grouped_mol_inits.values()))

    def _refine_coords_batch(
        self,
        mols_init: list,
        codes: list,
        maxiter: int = 10,
        batch_size_refinement: int = 100,
        fabric=None,
    ) -> torch.Tensor:
        """
        Refines the coordinates of molecules in batches.

        Args:
            mols_init (list): List of initial molecule data, where each element is a dictionary containing
                              'coords' and 'atoms_channel' tensors.
            codes (list): List of codes corresponding to each molecule.
            maxiter (int, optional): Maximum number of iterations for the optimizer. Default is 10.
            batch_size_refinement (int, optional): Number of molecules to process in each batch. Default is 100.
            fabric (optional): Fabric object that provides device and optimizer setup.

        Returns:
            torch.Tensor: Refined coordinates of all molecules concatenated along the first dimension.
        """
        num_batches = len(mols_init) // batch_size_refinement
        if len(mols_init) % batch_size_refinement != 0:
            nb_iter = num_batches + 1
        else:
            nb_iter = num_batches
        refined_coords = []

        for i in range(nb_iter):
            min_bound = i * batch_size_refinement
            max_bound = (len(mols_init) if i == num_batches else (i + 1) * batch_size_refinement)
            coords = torch.stack(
                [mols_init[j]["coords"].squeeze(0) for j in range(min_bound, max_bound)], dim=0,
            ).to(fabric.device)
            coords_init = coords.clone()
            coords.requires_grad = True

            with torch.no_grad():
                atoms_channel = torch.stack(
                    [mols_init[j]["atoms_channel"] for j in range(min_bound, max_bound)], dim=0,
                ).to(fabric.device)
                occupancy = (
                    torch.nn.functional.one_hot(atoms_channel.long(), self.n_channels).float().squeeze()
                ).to(fabric.device)
                code = torch.stack(
                    [codes[j] for j in range(min_bound, max_bound)], dim=0
                ).to(fabric.device)

            def closure():
                optimizer.zero_grad()
                pred = self.net(coords, code)
                loss = -(pred * occupancy).max(dim=2).values.mean()
                loss.backward()
                return loss

            optim_factory = partial(
                torch.optim.LBFGS,
                history_size=10,
                max_iter=4,
                line_search_fn="strong_wolfe",
                lr=1.0,
            )
            optimizer = fabric.setup_optimizers(optim_factory([coords]))
            tol, loss = 1e-4, 1e10
            for _ in range(maxiter):
                prev_loss = loss
                loss = optimizer.step(closure)
                if abs(loss - prev_loss).item() < tol:
                    break
                if (coords - coords_init).abs().max() > 1:
                    print("Refine coords diverges, so use initial coordinates...")
                    coords = coords_init
                    break
            refined_coords.append(coords.detach().cpu())
        return torch.cat(refined_coords, dim=0)

    def set_code_stats(self, code_stats: dict) -> None:
        """
        Set the code statistics.

        Args:
            code_stats: Code statistics.
        """
        self.code_stats = code_stats

    def unnormalize_code(self, codes: torch.Tensor):
        """
        Unnormalizes the given codes based on the provided normalization parameters.
        NOTE: This function is deprecated and will be removed in future versions.
        For now, it returns the original codes without unnormalization.

        Args:
            codes (torch.Tensor): The codes to be unnormalized.
            code_stats (dict): The statistics for the codes.

        Returns:
            torch.Tensor: The original codes (no unnormalization applied).
        """
        # Return original codes without unnormalization
        return codes


########################################################################################
## auxiliary functions
def local_maxima(data, order=1):
    size = 1 + 2 * order
    footprint = np.ones((size, size, size))
    footprint[order, order, order] = 0

    filtered = ndi.maximum_filter(data.numpy(), footprint=footprint)
    filtered = torch.from_numpy(filtered)
    data[data <= filtered] = 0
    return data


def find_peaks(voxel):
    voxel = voxel.squeeze()
    voxel[voxel < 0.25] = 0
    return torch.cat(
        [
            local_maxima(voxel[channel_idx], 1).unsqueeze(0)
            for channel_idx in range(voxel.shape[0])
        ],
        dim=0,
    )


def get_atom_coords(grid, rad=0.5):
    peaks = find_peaks(grid)
    # current version only works for fixed radius (ie, all atoms with same radius rad)
    coords = []
    atoms_channel = []
    radius = []

    for channel_idx in range(peaks.shape[0]):
        px, py, pz = torch.where(peaks[channel_idx] > 0)
        if px.numel() > 0:
            px, py, pz = px.float(), py.float(), pz.float()
            coords.append(torch.stack([px, py, pz], dim=1))
            atoms_channel.append(
                torch.full((px.shape[0],), channel_idx, dtype=torch.float32)
            )
            radius.append(torch.full((px.shape[0],), rad, dtype=torch.float32))

    if not coords:
        return None

    structure = {
        "coords": torch.cat(coords, dim=0).unsqueeze(0),
        "atoms_channel": torch.cat(atoms_channel, dim=0).unsqueeze(0),
        "radius": torch.cat(radius, dim=0).unsqueeze(0),
    }

    return structure


def _normalize_coords(mol, grid_dim):
    """
    NOTE: This function is deprecated and will be removed in future versions.
    For now, it returns the original molecule without normalization.
    """
    # Return original molecule without normalization
    return mol


def _unnormalize_coords(mol, grid_dim, resolution=0.25):
    """
    NOTE: This function is deprecated and will be removed in future versions.
    For now, it returns the original molecule without unnormalization.
    """
    # Return original molecule without unnormalization
    return mol


def get_grid(grid_dim, resolution=0.25):
    """
    Create a grid based on real-world distances.
    
    Args:
        grid_dim (int): Number of grid points per dimension
        resolution (float): Distance between grid points in Angstroms (default: 0.25)
    
    Returns:
        tuple: (discrete_grid, full_grid) where discrete_grid is 1D array and full_grid is 3D coordinates
    """
    # Calculate the total span in Angstroms
    total_span = (grid_dim - 1) * resolution
    half_span = total_span / 2
    
    # Create grid points in real space (Angstroms)
    discrete_grid = np.linspace(-half_span, half_span, grid_dim)
    
    # Create full 3D grid
    full_grid = torch.Tensor(
        [[a, b, c] for a in discrete_grid for b in discrete_grid for c in discrete_grid]
    )
    return discrete_grid, full_grid