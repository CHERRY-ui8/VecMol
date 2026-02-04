import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
from sklearn.cluster import DBSCAN
import torch.nn.functional as F
from vecmol.utils.constants import PADDING_INDEX, ELEMENTS_HASH_INV
import gc
import time

# Import split-out modules
from vecmol.utils.gnf_converter_modules import (
    ClusteringIterationRecord,
    ClusteringHistory,
    BondValidator,
    ConnectivityAnalyzer,
    ClusteringHistorySaver,
    ReconstructionMetrics,
    GradientFieldComputer,
    GradientAscentOptimizer,
    ClusteringProcessor,
    SamplingProcessor,
)


class GNFConverter(nn.Module):
    """
    Converter between molecular structures and Gradient Neural Fields (GNF).
    
    The GNF is defined as r = f(X,z) where:
    - X is the set of atomic coordinates
    - z is a query point in 3D space
    - r is the gradient at point z
    """
    
    def __init__(self,
                sigma: float,
                n_query_points: int,
                n_iter: int,
                step_size: float,
                eps: float,  # DBSCAN neighborhood radius
                min_samples: int,  # DBSCAN min samples
                sigma_ratios: Dict[str, float],
                gradient_field_method: str = "softmax",  # gradient field method: "gaussian", "softmax", "logsumexp", etc.
                temperature: float = 1.0,  # softmax temperature (sharper distribution when higher)
                logsumexp_eps: float = 1e-8,  # logsumexp numerical stability
                inverse_square_strength: float = 1.0,  # inverse-square strength
                gradient_clip_threshold: float = 0.3,  # gradient magnitude clip threshold
                sig_sf: float = 0.1,  # softmax field sigma
                sig_mag: float = 0.45,  # magnitude sigma
                gaussian_hole_clip: float = 0.8,  # gaussian_hole distance clip upper bound
                gradient_sampling_candidate_multiplier: int = 3,  # gradient sampling candidate multiplier
                field_variance_k_neighbors: int = 10,  # k-neighbors for field variance
                field_variance_weight: float = 1.0,  # field variance weight in sampling (softmax temperature)
                n_atom_types: int = 5,  # number of atom types (default 5 for backward compatibility)
                enable_early_stopping: bool = True,  # enable early stopping
                convergence_threshold: float = 1e-6,  # convergence threshold (gradient magnitude change)
                min_iterations: int = 50,  # minimum iterations before early stopping
                n_query_points_per_type: Optional[Dict[str, int]] = None,  # query_points per atom type; None = unified n_query_points
                enable_autoregressive_clustering: bool = False,  # enable autoregressive clustering
                initial_min_samples: Optional[int] = None,  # initial min_samples (default self.min_samples)
                min_samples_decay_factor: float = 0.7,  # min_samples decay per round
                min_min_samples: int = 2,  # min_samples lower bound
                max_clustering_iterations: int = 10,  # max clustering iterations
                bond_length_tolerance: float = 0.4,  # bond length upper tolerance (Å)
                bond_length_lower_tolerance: float = 0.2,  # bond length lower tolerance (Å)
                enable_clustering_history: bool = False,  # record clustering history
                debug_bond_validation: bool = False,  # debug bond validation output
                gradient_batch_size: Optional[int] = None,  # gradient batch size; None = process all at once
                n_initial_atoms_no_bond_check: int = 3,  # first N atoms exempt from bond length check
                enable_bond_validation: bool = True,  # enable bond length check
                sampling_range_min: float = -7.0,  # sampling range min (Å)
                sampling_range_max: float = 7.0):  # sampling range max (Å)
        super().__init__()
        self.sigma = sigma
        self.n_query_points = n_query_points
        self.n_iter = n_iter
        self.step_size = step_size
        self.eps = eps
        self.min_samples = min_samples
        self.sigma_ratios = sigma_ratios
        self.gradient_field_method = gradient_field_method
        self.temperature = temperature
        self.logsumexp_eps = logsumexp_eps
        self.inverse_square_strength = inverse_square_strength
        self.gradient_clip_threshold = gradient_clip_threshold
        self.sig_sf = sig_sf
        self.sig_mag = sig_mag
        self.gaussian_hole_clip = gaussian_hole_clip
        self.gradient_sampling_candidate_multiplier = gradient_sampling_candidate_multiplier
        self.field_variance_k_neighbors = field_variance_k_neighbors
        self.field_variance_weight = field_variance_weight
        self.n_atom_types = n_atom_types
        self.enable_early_stopping = enable_early_stopping
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations

        # Autoregressive clustering parameters
        self.enable_autoregressive_clustering = enable_autoregressive_clustering
        self.initial_min_samples = initial_min_samples
        self.min_samples_decay_factor = min_samples_decay_factor
        self.min_min_samples = min_min_samples
        self.max_clustering_iterations = max_clustering_iterations
        self.bond_length_tolerance = bond_length_tolerance
        self.bond_length_lower_tolerance = bond_length_lower_tolerance
        self.enable_clustering_history = enable_clustering_history
        self.debug_bond_validation = debug_bond_validation
        self.gradient_batch_size = gradient_batch_size
        self.n_initial_atoms_no_bond_check = n_initial_atoms_no_bond_check
        self.enable_bond_validation = enable_bond_validation
        self.sampling_range_min = sampling_range_min
        self.sampling_range_max = sampling_range_max
        
        # Per-atom-type sigma (5 for QM9, 6 for CREMP, 8 for GEOM-drugs)
        # Atom type index: 0=C, 1=H, 2=O, 3=N, 4=F, 5=S, 6=Cl, 7=Br
        atom_type_mapping = {0: 'C', 1: 'H', 2: 'O', 3: 'N', 4: 'F', 5: 'S', 6: 'Cl', 7: 'Br'}
        self.sigma_params = {}
        for atom_idx in range(n_atom_types):
            atom_symbol = atom_type_mapping.get(atom_idx, f'Type{atom_idx}')
            ratio = self.sigma_ratios.get(atom_symbol, 1.0)
            self.sigma_params[atom_idx] = sigma * ratio

        # Per-atom-type query_points; if n_query_points_per_type given use it, else unified n_query_points
        self.n_query_points_per_type = {}
        if n_query_points_per_type is not None:
            for atom_idx in range(n_atom_types):
                atom_symbol = atom_type_mapping.get(atom_idx, f'Type{atom_idx}')
                self.n_query_points_per_type[atom_idx] = n_query_points_per_type.get(atom_symbol, n_query_points)
        else:
            for atom_idx in range(n_atom_types):
                self.n_query_points_per_type[atom_idx] = n_query_points

        # Initialize split-out modules
        self.bond_validator = BondValidator(
            bond_length_tolerance=bond_length_tolerance,
            bond_length_lower_tolerance=bond_length_lower_tolerance,
            debug=debug_bond_validation
        )
        self.connectivity_analyzer = ConnectivityAnalyzer(bond_validator=self.bond_validator)
        self.history_saver = ClusteringHistorySaver(n_atom_types=n_atom_types)
        self.metrics_computer = ReconstructionMetrics()
        
        # Gradient field computer
        self.gradient_field_computer = GradientFieldComputer(
            sigma_params=self.sigma_params,
            sigma=sigma,
            gradient_field_method=gradient_field_method,
            temperature=temperature,
            logsumexp_eps=logsumexp_eps,
            inverse_square_strength=inverse_square_strength,
            gradient_clip_threshold=gradient_clip_threshold,
            sig_sf=sig_sf,
            sig_mag=sig_mag,
            gaussian_hole_clip=gaussian_hole_clip,
        )
        
        # Gradient ascent optimizer
        self.gradient_ascent_optimizer = GradientAscentOptimizer(
            n_iter=n_iter,
            step_size=step_size,
            sigma_params=self.sigma_params,
            sigma=sigma,
            enable_early_stopping=enable_early_stopping,
            convergence_threshold=convergence_threshold,
            min_iterations=min_iterations,
            gradient_batch_size=gradient_batch_size,
        )
        
        # Clustering processor
        self.clustering_processor = ClusteringProcessor(
            bond_validator=self.bond_validator,
            eps=eps,
            min_samples=min_samples,
            enable_autoregressive_clustering=enable_autoregressive_clustering,
            initial_min_samples=initial_min_samples,
            min_samples_decay_factor=min_samples_decay_factor,
            min_min_samples=min_min_samples,
            max_clustering_iterations=max_clustering_iterations,
            debug_bond_validation=debug_bond_validation,
            n_initial_atoms_no_bond_check=n_initial_atoms_no_bond_check,
            enable_bond_validation=enable_bond_validation,
        )
        
        # Sampling processor
        self.sampling_processor = SamplingProcessor(
            gradient_field_computer=self.gradient_field_computer,
            gradient_ascent_optimizer=self.gradient_ascent_optimizer,
            clustering_processor=self.clustering_processor,
            n_query_points=n_query_points,
            n_query_points_per_type=self.n_query_points_per_type,
            gradient_sampling_candidate_multiplier=gradient_sampling_candidate_multiplier,
            field_variance_k_neighbors=field_variance_k_neighbors,
            field_variance_weight=field_variance_weight,
            eps=eps,
            min_min_samples=min_min_samples,
            enable_clustering_history=enable_clustering_history,
            sampling_range_min=sampling_range_min,
            sampling_range_max=sampling_range_max,
        )
    
    def forward(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                query_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the GNF converter.
        """
        return self.mol2gnf(coords, atom_types, query_points)
    
    def mol2gnf(self, coords: torch.Tensor, atom_types: torch.Tensor, 
                query_points: torch.Tensor) -> torch.Tensor:
        """
        Convert coords and atom types to GNF (gradient neural field) vector field at query points.
        Gradient field points toward atoms so that gradient ascent moves points toward atoms.

        Args:
            coords: [batch, n_atoms, 3] or [n_atoms, 3]
            atom_types: [batch, n_atoms] or [n_atoms]
            query_points: [batch, n_points, 3] or [n_points, 3]

        Returns:
            vector_field: [batch, n_points, n_atom_types, 3]; batch dimension always kept.
        """
        # Handle batch dimension
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # [1, N, 3]
        if atom_types.dim() == 1:
            atom_types = atom_types.unsqueeze(0)  # [1, N]
        if query_points.dim() == 2:
            query_points = query_points.unsqueeze(0)  # [1, M, 3]
            
        n_atom_types = self.n_atom_types
        batch_size, n_points, _ = query_points.shape
        device = query_points.device
        vector_field = torch.zeros(batch_size, n_points, n_atom_types, 3, device=device)

        if batch_size == 1:
            mask = (atom_types[0] != PADDING_INDEX)  # [n_atoms]
            valid_coords = coords[0][mask]  # [n_valid_atoms, 3]
            valid_types = atom_types[0][mask].long()  # [n_valid_atoms]
            
            if valid_coords.size(0) > 0:
                vector_field[0] = self.gradient_field_computer.compute_gradient_field_matrix(
                    valid_coords, valid_types, query_points[0], n_atom_types
                )
        else:
            valid_data = []
            valid_indices = []
            
            for b in range(batch_size):
                mask = (atom_types[b] != PADDING_INDEX)  # [n_atoms]
                valid_coords = coords[b][mask]  # [n_valid_atoms, 3]
                valid_types = atom_types[b][mask].long()  # [n_valid_atoms]
                
                if valid_coords.size(0) > 0:
                    valid_data.append((valid_coords, valid_types, query_points[b]))
                    valid_indices.append(b)
            
            for idx, (valid_coords, valid_types, q_points) in zip(valid_indices, valid_data):
                vector_field[idx] = self.gradient_field_computer.compute_gradient_field_matrix(
                    valid_coords, valid_types, q_points, n_atom_types
                )
        
        return vector_field


    def gnf2mol(self, decoder: nn.Module, codes: torch.Tensor,
                _atom_types: Optional[torch.Tensor] = None,
                save_interval: Optional[int] = None,
                visualization_callback: Optional[callable] = None,
                predictor: Optional[nn.Module] = None,
                save_clustering_history: bool = False,
                clustering_history_dir: Optional[str] = None,
                save_gradient_ascent_sdf: bool = False,
                gradient_ascent_sdf_dir: Optional[str] = None,
                gradient_ascent_sdf_interval: int = 100,
                sample_id: Optional[Union[int, str]] = None,
                enable_timing: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reconstruct molecule coordinates from gradient field.
        Args:
            decoder: Decoder for vector field
            codes: [batch, grid_size**3, code_dim] (encoder output)
            _atom_types: Optional (unused, for API compatibility)
            save_interval: Optional; call visualization_callback every save_interval steps
            visualization_callback: (iteration_idx, all_points_dict, batch_idx); all_points_dict: type_idx -> [n_points, 3]
            predictor: Optional element existence predictor to filter absent types
            enable_timing: If True, print per-batch timing
        Returns:
            (final_coords, final_types)
        """
        t_global_start = time.perf_counter() if enable_timing else None

        device = codes.device
        batch_size = codes.size(0)
        n_atom_types = self.n_atom_types

        all_coords = []
        all_types = []
        
        # Process each batch (clustering/connectivity make full batching complex; process_atom_types_matrix uses batched decoder internally)
        for b in range(batch_size):
            t_batch_start = time.perf_counter() if enable_timing else None

            file_identifier = sample_id if sample_id is not None else b

            if b >= codes.size(0):
                break

            current_codes = codes[b:b+1]
            
            if current_codes.numel() == 0:
                continue
            
            if torch.isnan(current_codes).any() or torch.isinf(current_codes).any():
                continue
            
            t_predictor_start = time.perf_counter() if enable_timing else None
            element_existence = None
            if predictor is not None:
                with torch.no_grad():
                    logits = predictor(current_codes)  # [1, n_atom_types]
                    element_existence = torch.sigmoid(logits)  # [1, n_atom_types]
            t_predictor_end = time.perf_counter() if enable_timing else None
            
            iteration_callback = None
            if save_interval is not None and visualization_callback is not None:
                def create_iteration_callback(batch_idx, n_atom_types, save_interval_val, n_iter_val):
                    def callback(iter_idx, current_points, atom_types):
                        if iter_idx % save_interval_val == 0 or iter_idx == n_iter_val - 1:
                            all_points_dict = {}
                            for t in range(n_atom_types):
                                type_mask = (atom_types == t)
                                if type_mask.any():
                                    all_points_dict[t] = current_points[type_mask].clone()
                                else:
                                    all_points_dict[t] = torch.empty((0, 3), device=current_points.device)
                            visualization_callback(iter_idx, all_points_dict, batch_idx)
                    return callback
                
                iteration_callback = create_iteration_callback(b, n_atom_types, save_interval, self.n_iter)
            
            gradient_ascent_callback = None
            if save_gradient_ascent_sdf and gradient_ascent_sdf_dir:
                from pathlib import Path
                sdf_dir = Path(gradient_ascent_sdf_dir)
                sdf_dir.mkdir(parents=True, exist_ok=True)

                def create_gradient_ascent_sdf_callback(file_id, n_atom_types, interval):
                    def callback(iter_idx, current_points, atom_types):
                        if iter_idx % interval == 0 or iter_idx == self.n_iter - 1:
                            try:
                                from vecmol.sample_diffusion import xyz_to_sdf
                                elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(n_atom_types)]
                                
                                all_points = []
                                all_types = []
                                type_counts = {}
                                
                                for t in range(n_atom_types):
                                    type_mask = (atom_types == t)
                                    if not type_mask.any():
                                        continue
                                    
                                    type_points = current_points[type_mask].cpu().numpy()
                                    all_points.append(type_points)
                                    all_types.append(np.full(len(type_points), t))
                                    
                                    atom_symbol = elements[t] if t < len(elements) else f"Type{t}"
                                    type_counts[atom_symbol] = len(type_points)
                                
                                if len(all_points) > 0:
                                    combined_points = np.vstack(all_points)
                                    combined_types = np.concatenate(all_types)
                                    
                                    sdf_str = xyz_to_sdf(combined_points, combined_types, elements)
                                    lines = sdf_str.split('\n')
                                    type_info = ", ".join([f"{symbol}:{count}" for symbol, count in type_counts.items()])
                                    title = f"Gradient Ascent Iter {iter_idx}, Total {len(combined_points)} query_points ({type_info})"
                                    title = title[:80].ljust(80)
                                    lines[0] = title
                                    if isinstance(file_id, int):
                                        sdf_file = sdf_dir / f"sample_{file_id:04d}_iter_{iter_idx:04d}.sdf"
                                    else:
                                        sdf_file = sdf_dir / f"{file_id}_iter_{iter_idx:04d}.sdf"
                                    with open(sdf_file, 'w') as f:
                                        f.write('\n'.join(lines))
                            except Exception as e:
                                print(f"Warning: Failed to save gradient ascent SDF: {e}")
                    
                    return callback
                
                gradient_ascent_callback = create_gradient_ascent_sdf_callback(file_identifier, n_atom_types, gradient_ascent_sdf_interval)
            
            t_process_start = time.perf_counter() if enable_timing else None
            coords_list, types_list, histories = self.sampling_processor.process_atom_types_matrix(
                current_codes, n_atom_types, device=device, decoder=decoder,
                iteration_callback=iteration_callback,
                element_existence=element_existence,
                gradient_ascent_callback=gradient_ascent_callback,
                enable_timing=enable_timing
            )
            t_process_end = time.perf_counter() if enable_timing else None
            
            t_history_start = time.perf_counter() if enable_timing else None
            if save_clustering_history and clustering_history_dir and histories:
                self.history_saver.save_clustering_history(
                    histories, 
                    clustering_history_dir,
                    batch_idx=file_identifier
                )
            t_history_end = time.perf_counter() if enable_timing else None
            
            t_merge_start = time.perf_counter() if enable_timing else None
            t_connected_start = None
            t_connected_end = None
            if coords_list:
                merged_coords = torch.cat(coords_list, dim=0)
                merged_types = torch.cat(types_list, dim=0)

                if self.enable_bond_validation:
                    t_connected_start = time.perf_counter() if enable_timing else None
                    filtered_coords, filtered_types = self.connectivity_analyzer.select_largest_connected_component(
                        merged_coords, merged_types
                    )
                    t_connected_end = time.perf_counter() if enable_timing else None
                else:
                    valid_mask = (merged_types != PADDING_INDEX) & (merged_types != -1)
                    if valid_mask.any():
                        filtered_coords = merged_coords[valid_mask]
                        filtered_types = merged_types[valid_mask]
                    else:
                        filtered_coords = torch.empty(0, 3, device=device)
                        filtered_types = torch.empty(0, dtype=torch.long, device=device)
                    t_connected_start = None
                    t_connected_end = None
                
                all_coords.append(filtered_coords)
                all_types.append(filtered_types)
            else:
                all_coords.append(torch.empty(0, 3, device=device))
                all_types.append(torch.empty(0, dtype=torch.long, device=device))
            t_merge_end = time.perf_counter() if enable_timing else None

            if enable_timing:
                t_batch_end = time.perf_counter()
                predictor_time = (t_predictor_end - t_predictor_start) if (t_predictor_start is not None and t_predictor_end is not None) else 0.0
                process_time = (t_process_end - t_process_start) if (t_process_start is not None and t_process_end is not None) else 0.0
                history_time = (t_history_end - t_history_start) if (t_history_start is not None and t_history_end is not None) else 0.0
                merge_time = (t_merge_end - t_merge_start) if (t_merge_start is not None and t_merge_end is not None) else 0.0
                connected_time = (t_connected_end - t_connected_start) if (t_connected_start is not None and t_connected_end is not None) else 0.0
                total_batch_time = t_batch_end - t_batch_start if t_batch_start is not None else 0.0
                print(
                    f"[GNFConverter.gnf2mol] Batch {file_identifier} timing: "
                    f"predictor={predictor_time:.3f}s, "
                    f"process={process_time:.3f}s, "
                    f"merge={merge_time:.3f}s, "
                    f"connected_component={connected_time:.3f}s, "
                    f"history_io={history_time:.3f}s, "
                    f"total={total_batch_time:.3f}s"
                )
        
        # Pad to batch max length
        max_atoms = max([c.size(0) for c in all_coords]) if all_coords else 0
        final_coords = torch.stack([F.pad(c, (0,0,0,max_atoms-c.size(0))) if c.size(0)<max_atoms else c for c in all_coords], dim=0)
        final_types = torch.stack([F.pad(t, (0,max_atoms-t.size(0)), value=-1) if t.size(0)<max_atoms else t for t in all_types], dim=0)

        if enable_timing and t_global_start is not None:
            t_global_end = time.perf_counter()
            print(f"[GNFConverter.gnf2mol] Total time for all batches: {t_global_end - t_global_start:.3f}s")

        return final_coords, final_types # [batch, n_atoms, 3]



    def compute_reconstruction_metrics(
        self,
        recon_coords: torch.Tensor,
        recon_types: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_types: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction metrics between reconstructed and ground-truth molecules.

        Args:
            recon_coords: [batch_size, n_atoms, 3]
            recon_types: [batch_size, n_atoms]
            gt_coords: [batch_size, n_atoms, 3]
            gt_types: [batch_size, n_atoms]

        Returns:
            Dict of metric names to values.
        """
        return self.metrics_computer.compute_reconstruction_metrics(
            recon_coords, recon_types, gt_coords, gt_types
        )
    