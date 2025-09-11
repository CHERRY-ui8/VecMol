import math

import torch
from torch import nn
from tqdm import tqdm
from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.utils.utils_nf import unnormalize_code
from funcmol.models.unet1d import MLPResCode
from funcmol.models.egnn_denoiser import GNNDenoiser


########################################################################################
# Helper function for printing
def _print(fabric, message):
    """Print message using fabric if available, otherwise use regular print."""
    if fabric is not None:
        fabric.print(message)
    else:
        print(message)

########################################################################################
# create funcmol
def create_funcmol(config: dict, fabric: object = None):
    """
    Create and compile a FuncMol model.

    Args:
        config (dict): Configuration dictionary for the FuncMol model.
        fabric (object): An object providing necessary methods and attributes for model creation.
                        If None, will use single GPU mode.

    Returns:
        torch.nn.Module: The compiled FuncMol model.
    """
    model = FuncMol(config, fabric=fabric)

    # n params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _print(fabric, f">> FuncMol has {(n_params/1e6):.02f}M parameters")

    # Disable torch.compile due to compatibility issues with torch_cluster
    # model = torch.compile(model)

    return model


########################################################################################
# FuncMol class
class FuncMol(nn.Module):
    def __init__(self, config: dict, fabric=None):
        super().__init__()
        # Handle both fabric and single GPU modes
        if fabric is not None:
            self.device = fabric.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.smooth_sigma = config["smooth_sigma"]
        self.grid_size = config["dset"]["grid_size"]  # 保存grid_size配置

        # 根据配置选择denoiser类型
        if config.get("denoiser", {}).get("use_gnn", False):
            # 使用GNN denoiser
            self.net = GNNDenoiser(
                code_dim=config["decoder"]["code_dim"],
                hidden_dim=config["denoiser"]["n_hidden_units"],
                num_blocks=config["denoiser"]["num_blocks"],
                k_neighbors=config["denoiser"]["k_neighbors"],
                cutoff=config["denoiser"]["cutoff"],
                radius=config["denoiser"]["radius"],
                dropout=config["denoiser"]["dropout"],
                grid_size=config["dset"]["grid_size"],
                anchor_spacing=config["dset"]["anchor_spacing"],
                use_radius_graph=config["denoiser"]["use_radius_graph"],
                device=self.device
            )
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

    def forward(self, y: torch.Tensor, debug: bool = False):
        """
        Forward pass of the denoiser model.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, channel_size, c_size).
            debug (bool): Whether to print debug information.

        Returns:
            torch.Tensor: Output tensor after passing through the denoiser model.
        """
        if debug:
            print(f"[DENOISER DEBUG] Input y - min: {y.min().item():.6f}, max: {y.max().item():.6f}, mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")
            print(f"[DENOISER DEBUG] Input y - shape: {y.shape}, device: {y.device}, dtype: {y.dtype}")
        
        xhat = self.net(y)
        
        if debug:
            print(f"[DENOISER DEBUG] Output xhat - min: {xhat.min().item():.6f}, max: {xhat.max().item():.6f}, mean: {xhat.mean().item():.6f}, std: {xhat.std().item():.6f}")
            print(f"[DENOISER DEBUG] Output xhat - shape: {xhat.shape}")
            
            # 检查是否有异常值
            if torch.isnan(xhat).any():
                print("[DENOISER DEBUG] WARNING: xhat contains NaN values!")
            if torch.isinf(xhat).any():
                print("[DENOISER DEBUG] WARNING: xhat contains Inf values!")
            
            # 检查输出范围是否合理
            if xhat.abs().max().item() > 100.0:
                print(f"[DENOISER DEBUG] WARNING: xhat has very large values (max abs: {xhat.abs().max().item():.2f})")
        
        return xhat

    def score(self, y: torch.Tensor, debug: bool = False):
        """
        Calculates the score of the denoiser model.

        Args:
        - y: Input tensor of shape (batch_size, channels, height, width).
        - debug (bool): Whether to print debug information.

        Returns:
        - score: The score tensor of shape (batch_size, channels, height, width).
        """
        xhat = self.forward(y, debug=debug)
        score = (xhat - y) / (self.smooth_sigma**2)
        
        if debug:
            print(f"[SCORE DEBUG] xhat - y difference - min: {(xhat-y).min().item():.6f}, max: {(xhat-y).max().item():.6f}, mean: {(xhat-y).mean().item():.6f}")
            print(f"[SCORE DEBUG] smooth_sigma: {self.smooth_sigma}, smooth_sigma^2: {self.smooth_sigma**2}")
            print(f"[SCORE DEBUG] Raw score - min: {score.min().item():.6f}, max: {score.max().item():.6f}, mean: {score.mean().item():.6f}, std: {score.std().item():.6f}")
        
        # 添加数值稳定性：梯度裁剪
        max_score_norm = 10.0  # 可调参数
        score_norm = torch.norm(score, dim=-1, keepdim=True)
        max_norm = score_norm.max().item()
        
        if debug:
            print(f"[SCORE DEBUG] Score norm - max: {max_norm:.6f}, mean: {score_norm.mean().item():.6f}")
        
        if max_norm > max_score_norm:
            if debug:
                print(f"[SCORE DEBUG] Clipping score norm from {max_norm:.6f} to {max_score_norm:.6f}")
            score = torch.where(
                score_norm > max_score_norm,
                score * (max_score_norm / score_norm),
                score
            )
        
        if debug:
            final_norm = torch.norm(score, dim=-1, keepdim=True).max().item()
            print(f"[SCORE DEBUG] Final score - min: {score.min().item():.6f}, max: {score.max().item():.6f}, norm: {final_norm:.6f}")
        
        return score

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
        temperature: float = 1.0,
        debug: bool = False
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
            for step in range(n_steps):
                q += delta * p / 2  # q_{t+1/2}
                psi = self.score(q, debug=debug and step % 20 == 0)  # 每20步打印一次debug信息
                p += u * delta * psi / 2  # p_{t+1}
                p = (
                    zeta1 * p + u * delta * psi / 2 + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)
                )  # p_{t+1}
                q += delta * p / 2  # q_{t+1}
                
                # 数值稳定性检查：每10步检查一次数值范围
                if step % 10 == 0:
                    q_norm = torch.norm(q, dim=-1).max().item()
                    p_norm = torch.norm(p, dim=-1).max().item()
                    if debug:
                        print(f"[WJS DEBUG] Step {step}: q_norm={q_norm:.4f}, p_norm={p_norm:.4f}")
                    if q_norm > 100.0 or p_norm > 100.0:
                        if debug:
                            print(f"[WJS DEBUG] Clipping large values at step {step}")
                        # 如果数值过大，进行裁剪
                        q = torch.clamp(q, -50.0, 50.0)
                        p = torch.clamp(p, -50.0, 50.0)
        elif scheme == "baoab":
            for step in range(n_steps):
                p += u * delta * self.score(q, debug=debug and step % 20 == 0) / 2  # p_{t+1/2}
                q += delta * p / 2  # q_{t+1/2}
                phat = zeta1 * p + math.sqrt(temperature) * math.sqrt(u * (1 - zeta2)) * torch.randn_like(q)  # phat_{t+1/2}
                q += delta * phat / 2  # q_{t+1}
                psi = self.score(q, debug=debug and step % 20 == 0)
                p = phat + u * delta * psi / 2  # p_{t+1}
                
                # 数值稳定性检查：每10步检查一次数值范围
                if step % 10 == 0:
                    q_norm = torch.norm(q, dim=-1).max().item()
                    p_norm = torch.norm(p, dim=-1).max().item()
                    if debug:
                        print(f"[WJS DEBUG] Step {step}: q_norm={q_norm:.4f}, p_norm={p_norm:.4f}")
                    if q_norm > 100.0 or p_norm > 100.0:
                        if debug:
                            print(f"[WJS DEBUG] Clipping large values at step {step}")
                        # 如果数值过大，进行裁剪
                        q = torch.clamp(q, -50.0, 50.0)
                        p = torch.clamp(p, -50.0, 50.0)
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

        return y, torch.zeros_like(y)

    def sample(
        self,
        config: dict,
        fabric = None,
        delete_net: bool = False,
        code_stats: dict = None,  # noqa: ARG002
        debug: bool = False,
    ):
        """
        Samples molecular modulation codes using the Walk-Jump-Sample (WJS) method.

        Args:
            config (dict): Configuration dictionary containing parameters for WJS.
            fabric (optional): Fabric object for printing and logging.
            delete_net (bool, optional): If True, deletes the network and clears CUDA cache after sampling. Default is False.
            code_stats (dict, optional): Code statistics for unnormalization. If provided, codes will be unnormalized before returning.

        Returns:
            torch.Tensor: Generated codes tensor of shape (n_samples, n_grid, code_dim).
        """
        self.eval()

        #  sample codes with WJS
        codes_all = []
        _print(fabric, f">> Sample codes with WJS (n_chains: {config['wjs']['n_chains']})")
        try:
            for _ in tqdm(range(config["wjs"]["repeats_wjs"])):
                # initialize y and v
                y, v = self.initialize_y_v(
                    n_chains=config["wjs"]["n_chains"],
                    code_dim=config["decoder"]["code_dim"],
                )

                # walk and jump
                for step_idx in range(0, config["wjs"]["max_steps_wjs"], config["wjs"]["steps_wjs"]):
                    # 记录walk前的数值范围
                    y_norm_before = torch.norm(y, dim=-1).max().item()
                    v_norm_before = torch.norm(v, dim=-1).max().item()
                    
                    y, v = self.wjs_walk_steps(y, v, config["wjs"]["steps_wjs"], delta=config["wjs"]["delta_wjs"], friction=config["wjs"]["friction_wjs"], debug=debug)  # walk
                    
                    # 记录walk后的数值范围
                    y_norm_after = torch.norm(y, dim=-1).max().item()
                    v_norm_after = torch.norm(v, dim=-1).max().item()
                    
                    # 如果数值范围异常，打印警告
                    if y_norm_after > 100.0 or v_norm_after > 100.0:
                        _print(fabric, f">> WARNING: Large values detected at step {step_idx}")
                        _print(fabric, f">> y_norm: {y_norm_before:.2f} -> {y_norm_after:.2f}")
                        _print(fabric, f">> v_norm: {v_norm_before:.2f} -> {v_norm_after:.2f}")
                    
                    code_hats = self.wjs_jump_step(y)  # jump
                    codes_all.append(code_hats.cpu())
                    
                # 清理GPU内存
                del y, v
                torch.cuda.empty_cache()
                
        except Exception as e:
            _print(fabric, f"Error during WJS sampling: {e}")
            raise e

        # Clean up network if requested
        if delete_net:
            del self.net
            torch.cuda.empty_cache()

        # Concatenate all generated codes
        codes = torch.cat(codes_all, dim=0)
        _print(fabric, f">> Generated {codes.size(0)} codes")
        
        # 添加调试信息：检查codes的维度和内容
        _print(fabric, f">> Codes shape: {codes.shape}")
        _print(fabric, f">> Codes dtype: {codes.dtype}")
        _print(fabric, f">> Codes device: {codes.device}")
        
        # 检查codes是否包含NaN/Inf
        if torch.isnan(codes).any():
            _print(fabric, ">> WARNING: codes contains NaN values!")
            nan_count = torch.isnan(codes).sum().item()
            _print(fabric, f">> NaN count: {nan_count}")
        
        if torch.isinf(codes).any():
            _print(fabric, ">> WARNING: codes contains Inf values!")
            inf_count = torch.isinf(codes).sum().item()
            _print(fabric, f">> Inf count: {inf_count}")
        
        # 检查codes的数值范围（normalized）
        _print(fabric, f">> Codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")

        # 如果提供了code_stats，进行unnormalization
        if code_stats is not None:
            _print(fabric, ">> Unnormalizing codes...")
            codes = unnormalize_code(codes, code_stats)
            _print(fabric, f">> Unnormalized codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")
        else:
            _print(fabric, ">> No code_stats provided, returning normalized codes")

        # 清理内存
        del codes_all
        torch.cuda.empty_cache()

        return codes
