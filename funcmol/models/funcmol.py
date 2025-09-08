import math

import torch
from torch import nn
from tqdm import tqdm
from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.models.unet1d import MLPResCode
from funcmol.models.egnn import EGNNVectorField


########################################################################################
# Simple EGNN-based denoiser using functions
def create_egnn_denoiser(config, device):
    """
    Create a simple EGNN-based denoiser using EGNNVectorField.
    
    Args:
        config: Configuration dictionary
        device: Device to run on
        
    Returns:
        A simple denoiser function
    """
    # Create EGNNVectorField
    egnn = EGNNVectorField(
        grid_size=config["dset"]["grid_size"],
        hidden_dim=config["denoiser"]["n_hidden_units"],
        num_layers=config["denoiser"]["num_blocks"],
        radius=config["denoiser"]["radius"],
        n_atom_types=config["dset"]["n_channels"],
        code_dim=config["decoder"]["code_dim"],
        cutoff=config["denoiser"]["cutoff"],
        anchor_spacing=config["dset"]["anchor_spacing"],
        device=device,
        k_neighbors=config["denoiser"]["k_neighbors"]
    )
    
    # Simple MLP for output projection
    output_projection = nn.Sequential(
        nn.Linear(config["decoder"]["code_dim"], config["denoiser"]["n_hidden_units"]),
        nn.SiLU(),
        nn.Dropout(config["denoiser"]["dropout"]),
        nn.Linear(config["denoiser"]["n_hidden_units"], config["decoder"]["code_dim"])
    ).to(device)
    
    def denoiser_fn(y):
        """
        Simple denoiser function using EGNN.
        
        Args:
            y: Input codes of shape (batch_size, n_grid, code_dim)
            
        Returns:
            Denoised codes of the same shape
        """
        batch_size = y.size(0)
        n_grid = y.size(1)
        
        # Create dummy query points for EGNNVectorField
        dummy_query_points = torch.randn(batch_size, 1, 3, device=y.device)
        
        # Use EGNNVectorField to process the codes
        with torch.no_grad():
            _ = egnn(dummy_query_points, y)
        
        # Apply output projection
        y_flat = y.view(-1, config["decoder"]["code_dim"])
        denoised_flat = output_projection(y_flat)
        denoised = denoised_flat.view(batch_size, n_grid, config["decoder"]["code_dim"])
        
        return denoised
    
    return denoiser_fn


########################################################################################
# create funcmol
def create_funcmol(config: dict, fabric: object):
    """
    Create and compile a FuncMol model.

    Args:
        config (dict): Configuration dictionary for the FuncMol model.
        fabric (object): An object providing necessary methods and attributes for model creation.

    Returns:
        torch.nn.Module: The compiled FuncMol model.
    """
    model = FuncMol(config, fabric=fabric)

    # n params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fabric.print(f">> FuncMol has {(n_params/1e6):.02f}M parameters")

    # Disable torch.compile due to compatibility issues with torch_cluster
    # model = torch.compile(model)

    return model


########################################################################################
# FuncMol class
class FuncMol(nn.Module):
    def __init__(self, config: dict, fabric):
        super().__init__()
        self.device = fabric.device
        self.smooth_sigma = config["smooth_sigma"]
        self.grid_size = config["dset"]["grid_size"]  # 保存grid_size配置

        # 根据配置选择denoiser类型
        if config.get("denoiser", {}).get("use_gnn", False):
            # 使用EGNN denoiser function
            self.net = create_egnn_denoiser(config, self.device)
        else:
            # 使用MLP denoiser (默认)
            self.net = MLPResCode(
                code_dim=config["decoder"]["code_dim"],
                n_hidden_units=config["denoiser"]["n_hidden_units"],
                num_blocks=config["denoiser"]["num_blocks"],
                n_groups=config["denoiser"]["n_groups"],
                dropout=config["denoiser"]["dropout"],
                bias_free=config["denoiser"]["bias_free"],
            )

    def forward(self, y: torch.Tensor):
        """
        Forward pass of the denoiser model.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, channel_size, c_size).

        Returns:
            torch.Tensor: Output tensor after passing through the denoiser model.
        """
        return self.net(y)

    def score(self, y: torch.Tensor):
        """
        Calculates the score of the denoiser model.

        Args:
        - y: Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - score: The score tensor of shape (batch_size, channels, height, width).
        """
        xhat = self.forward(y)
        return (xhat - y) / (self.smooth_sigma**2)

    @torch.no_grad()
    def wjs_walk_steps(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        n_steps: int,
        delta: float = 0.5,
        friction: float = 1.0,
        lipschitz: float = 1.0,
        scheme: str = "aboba",
        temperature: float = 1.0
    ):
        """
        Perform a series of steps using the Weighted Jump Stochastic (WJS) method.

        Args:
            q (torch.Tensor): The initial position tensor.
            p (torch.Tensor): The initial momentum tensor.
            n_steps (int): The number of steps to perform.
            delta (float, optional): The time step size. Default is 0.5.
            friction (float, optional): The friction coefficient. Default is 1.0.
            lipschitz (float, optional): The Lipschitz constant. Default is 1.0.
            scheme (str, optional): The integration scheme to use, either "aboba" or "baoab". Default is "aboba".
            temperature (float, optional): The temperature parameter. Default is 1.0.

        Returns:
            tuple: A tuple containing the updated position tensor `q` and the updated momentum tensor `p`.
        """
        u = pow(lipschitz, -1)  # inverse mass
        delta *= self.smooth_sigma
        zeta1 = math.exp(-friction * delta)
        zeta2 = math.exp(-2 * friction * delta)
        if scheme == "aboba":
            for _ in range(n_steps):
                q += delta * p / 2  # q_{t+1/2}
                psi = self.score(q)
                p += u * delta * psi / 2  # p_{t+1}
                p = (
                    zeta1 * p + u * delta * psi / 2 + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)
                )  # p_{t+1}
                q += delta * p / 2  # q_{t+1}
        elif scheme == "baoab":
            for _ in range(n_steps):
                p += u * delta * self.score(q) / 2  # p_{t+1/2}
                q += delta * p / 2  # q_{t+1/2}
                phat = zeta1 * p + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)  # phat_{t+1/2}
                q += delta * phat / 2  # q_{t+1}
                psi = self.score(q)
                p = phat + u * delta * psi / 2  # p_{t+1}
        return q, p

    @torch.no_grad()
    def wjs_jump_step(self, y: torch.Tensor):
        """Jump step of walk-jump sampling.
        Recover clean sample x from noisy sample y.
        It is a simple forward of the network.

        Args:
            y (torch.Tensor): samples y from mcmc chain


        Returns:
            torch.Tensor: estimated ``clean'' samples xhats
        """
        return self.forward(y)

    def initialize_y_v(
        self,
        n_chains: int = 25,
        code_dim: int = 1024,
    ):
        """
        Initializes the latent variable `y` with uniform noise and adds Gaussian noise.

        Args:
            n_chains (int, optional): Number of chains to initialize. Defaults to 25.
            code_dim (int, optional): Dimensionality of the code. Defaults to 1024.

        Returns:
            tuple: A tuple containing:
                - y (torch.Tensor): Tensor of shape (n_chains, n_grid, code_dim) with added Gaussian noise.
                - torch.Tensor: Tensor of zeros with the same shape as `y`.
        """
        # 计算grid维度 - 使用保存的grid_size
        n_grid = self.grid_size ** 3
        
        # uniform noise - 生成包含grid维度的codes
        y = torch.empty((n_chains, n_grid, code_dim), device=self.device, dtype=torch.float32).uniform_(-1, 1)

        # gaussian noise
        y = add_noise_to_code(y, self.smooth_sigma)
        
        # 确保y在正确的设备上
        y = y.to(self.device)

        return y, torch.zeros_like(y)

    def sample(
        self,
        config: dict,
        fabric = None,
        delete_net: bool = False,
    ):
        """
        Samples molecular modulation codes using the Walk-Jump-Sample (WJS) method.

        Args:
            config (dict): Configuration dictionary containing parameters for WJS.
            fabric (optional): Fabric object for printing and logging.
            delete_net (bool, optional): If True, deletes the network and clears CUDA cache after sampling. Default is False.

        Returns:
            torch.Tensor: Generated codes tensor of shape (n_samples, n_grid, code_dim).
        """
        self.eval()

        #  sample codes with WJS
        codes_all = []
        fabric.print(f">> Sample codes with WJS (n_chains: {config['wjs']['n_chains']})")
        try:
            for _ in tqdm(range(config["wjs"]["repeats_wjs"])):
                # initialize y and v
                y, v = self.initialize_y_v(
                    n_chains=config["wjs"]["n_chains"],
                    code_dim=config["decoder"]["code_dim"],
                )

                # walk and jump
                for _ in range(0, config["wjs"]["max_steps_wjs"], config["wjs"]["steps_wjs"]):
                    y, v = self.wjs_walk_steps(y, v, config["wjs"]["steps_wjs"], delta=config["wjs"]["delta_wjs"], friction=config["wjs"]["friction_wjs"])  # walk
                    code_hats = self.wjs_jump_step(y)  # jump
                    codes_all.append(code_hats.cpu())
                    
                # 清理GPU内存
                del y, v
                torch.cuda.empty_cache()
                
        except Exception as e:
            fabric.print(f"Error during WJS sampling: {e}")
            raise e

        # Clean up network if requested
        if delete_net:
            del self.net
            torch.cuda.empty_cache()

        # Concatenate all generated codes
        codes = torch.cat(codes_all, dim=0)
        fabric.print(f">> Generated {codes.size(0)} codes")
        
        # 添加调试信息：检查codes的维度和内容
        fabric.print(f">> Codes shape: {codes.shape}")
        fabric.print(f">> Codes dtype: {codes.dtype}")
        fabric.print(f">> Codes device: {codes.device}")
        
        # 检查codes是否包含NaN/Inf
        if torch.isnan(codes).any():
            fabric.print(">> WARNING: codes contains NaN values!")
            nan_count = torch.isnan(codes).sum().item()
            fabric.print(f">> NaN count: {nan_count}")
        
        if torch.isinf(codes).any():
            fabric.print(">> WARNING: codes contains Inf values!")
            inf_count = torch.isinf(codes).sum().item()
            fabric.print(f">> Inf count: {inf_count}")
        
        # 检查codes的数值范围
        fabric.print(f">> Codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")

        # 清理内存
        del codes_all
        torch.cuda.empty_cache()

        return codes
