from typing import Optional

import torch
from torch import nn
from tqdm import tqdm
from vecmol.utils.utils_nf import unnormalize_code
from vecmol.models.egnn_denoiser import GNNDenoiser
from vecmol.models.ddpm import (create_diffusion_constants, compute_ddpm_loss,
    p_sample_loop, compute_ddpm_loss_x0, p_sample_loop_x0)


########################################################################################
# create vecmol
def create_vecmol(config: dict):
    """
    Create and compile a VecMol model.

    Args:
        config (dict): Configuration dictionary for the VecMol model.

    Returns:
        torch.nn.Module: The compiled VecMol model.
    """
    model = VecMol(config)

    # n params
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f">> VecMol has {(n_params/1e6):.02f}M parameters")

    # Disable torch.compile due to compatibility issues with torch_cluster
    # model = torch.compile(model)

    return model


########################################################################################
# VecMol class
class VecMol(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.smooth_sigma = config["smooth_sigma"]
        self.grid_size = config["dset"]["grid_size"]  # 保存grid_size配置

        # DDPM 预测目标：ddpm_epsilon（预测噪声）或 ddpm_x0（预测 x0）
        self.diffusion_method = config.get("diffusion_method", "ddpm_x0")

        # 统一从 denoiser 配置读取 GNN 模型参数
        denoiser_config = config.get("denoiser", {})
        self.net = GNNDenoiser(
            code_dim=config["decoder"]["code_dim"],
            hidden_dim=denoiser_config.get("hidden_dim", 128),
            num_layers=denoiser_config.get("num_layers", 4),
            time_emb_dim=denoiser_config.get("time_emb_dim", 64),
            cutoff=denoiser_config.get("cutoff", 5.0),
            radius=denoiser_config.get("radius", 2.0),
            dropout=denoiser_config.get("dropout", 0.1),
            grid_size=config["dset"]["grid_size"],
            anchor_spacing=config["dset"]["anchor_spacing"],
            use_radius_graph=denoiser_config.get("use_radius_graph", True),
            device=self.device
        ).to(self.device)

        self.diffusion_consts = create_diffusion_constants(config, device=self.device)
        ddpm_config = config.get("ddpm", {})
        self.num_timesteps = ddpm_config.get("num_timesteps", 1000)
        self.use_time_weight = ddpm_config.get("use_time_weight", True)

    def forward(self, y: torch.Tensor, t: torch.Tensor = None, debug: bool = False):
        """
        Forward pass of the denoiser model.

        Args:
            y (torch.Tensor): Input tensor of shape (batch_size, channel_size, c_size) or (batch_size, N, N, N, code_dim) for DDPM.
            t (torch.Tensor, optional): Time steps for DDPM. Required when using DDPM method.
            debug (bool): Whether to print debug information.

        Returns:
            torch.Tensor: Output tensor after passing through the denoiser model.
        """
        if debug:
            print(f"[DENOISER DEBUG] Input y - min: {y.min().item():.6f}, max: {y.max().item():.6f}, mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")
            print(f"[DENOISER DEBUG] Input y - shape: {y.shape}, device: {y.device}, dtype: {y.dtype}")

        if t is None:
            raise ValueError("Time steps 't' are required for DDPM")
        xhat = self.net(y, t)

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

    def score(self, y: torch.Tensor, t: torch.Tensor = None, debug: bool = False):
        """
        Calculates the score of the denoiser model.

        Args:
        - y: Input tensor of shape (batch_size, channels, height, width) or (batch_size, N, N, N, code_dim) for DDPM.
        - t: Time steps for DDPM. Required when using DDPM method.
        - debug (bool): Whether to print debug information.

        Returns:
        - score: The score tensor of shape (batch_size, channels, height, width).
        """
        if t is None:
            raise ValueError("Time steps 't' are required for DDPM")
        score = self.forward(y, t, debug=debug)

        if debug:
            print(f"[SCORE DEBUG] Score - min: {score.min().item():.6f}, max: {score.max().item():.6f}, mean: {score.mean().item():.6f}, std: {score.std().item():.6f}")

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

    def train_ddpm_step(self, x_0: torch.Tensor, position_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        DDPM训练步骤

        Args:
            x_0: 原始数据 [B, N*N*N, code_dim]
            position_weights: 位置权重 [B, N*N*N]，可选

        Returns:
            训练损失
        """
        if self.diffusion_method == "ddpm_epsilon":
            return compute_ddpm_loss(self.net, x_0, self.diffusion_consts, self.device, position_weights=position_weights)
        elif self.diffusion_method == "ddpm_x0":
            return compute_ddpm_loss_x0(self.net, x_0, self.diffusion_consts, self.device, use_time_weight=self.use_time_weight, position_weights=position_weights)
        else:
            raise ValueError(f"diffusion_method must be 'ddpm_epsilon' or 'ddpm_x0', got '{self.diffusion_method}'")

    @torch.no_grad()
    def sample_ddpm(self, shape: tuple, code_stats=None, progress: bool = True, clip_denoised: bool = False) -> torch.Tensor:
        """
        DDPM采样

        Args:
            shape: 输出形状 (batch_size, N*N*N, code_dim)
            progress: 是否显示进度条
            clip_denoised: 是否将去噪后的结果裁剪到合理范围（用于数值稳定性，默认False）

        Returns:
            生成的样本 [B, N*N*N, code_dim]
        """
        if self.diffusion_method == "ddpm_epsilon":
            self.net.eval()
            sampled = p_sample_loop(self.net, shape, self.diffusion_consts, self.device, progress, clip_denoised=clip_denoised)
        elif self.diffusion_method == "ddpm_x0":
            self.net.eval()
            sampled = p_sample_loop_x0(self.net, shape, self.diffusion_consts, self.device, progress, clip_denoised=clip_denoised)
        else:
            raise ValueError(f"diffusion_method must be 'ddpm_epsilon' or 'ddpm_x0', got '{self.diffusion_method}'")

        if code_stats is not None:
            sampled = unnormalize_code(sampled, code_stats)
        return sampled

    def sample(
        self,
        config: dict,
        delete_net: bool = False,
        code_stats: dict = None,  # noqa: ARG002
        debug: bool = False,
    ):
        """
        Samples molecular modulation codes using DDPM.

        Args:
            config (dict): Configuration dictionary containing parameters for sampling.
            delete_net (bool, optional): If True, deletes the network and clears CUDA cache after sampling. Default is False.
            code_stats (dict, optional): Code statistics for unnormalization. If provided, codes will be unnormalized before returning.

        Returns:
            torch.Tensor: Generated codes tensor of shape (n_samples, n_grid, code_dim).
        """
        self.eval()
        return self._sample_ddpm(config, delete_net, code_stats, debug)

    def _sample_ddpm(self, config: dict, delete_net: bool, code_stats: dict, debug: bool = False):  # noqa: ARG002
        """
        DDPM采样方法
        """
        codes_all = []
        n_samples = config.get("ddpm", {}).get("n_samples", 100)
        batch_size = config.get("ddpm", {}).get("batch_size", 10)

        print(f">> Sample codes with DDPM (n_samples: {n_samples})")

        try:
            for _ in tqdm(range(0, n_samples, batch_size)):
                current_batch_size = min(batch_size, n_samples - len(codes_all))

                # 定义DDPM采样形状 [batch_size, N*N*N, code_dim]
                shape = (current_batch_size, self.grid_size**3, config["decoder"]["code_dim"])

                # DDPM采样
                codes_batch = self.sample_ddpm(shape, progress=False)
                codes_all.append(codes_batch.cpu())

                # 清理GPU内存
                del codes_batch
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during DDPM sampling: {e}")
            raise e

        # Clean up network if requested
        if delete_net:
            del self.net
            torch.cuda.empty_cache()

        # Concatenate all generated codes
        codes = torch.cat(codes_all, dim=0)
        print(f">> Generated {codes.size(0)} codes")

        # 添加调试信息：检查codes的维度和内容
        print(f">> Codes shape: {codes.shape}")
        print(f">> Codes dtype: {codes.dtype}")
        print(f">> Codes device: {codes.device}")

        # 检查codes是否包含NaN/Inf
        if torch.isnan(codes).any():
            print(">> WARNING: codes contains NaN values!")
            nan_count = torch.isnan(codes).sum().item()
            print(f">> NaN count: {nan_count}")

        if torch.isinf(codes).any():
            print(">> WARNING: codes contains Inf values!")
            inf_count = torch.isinf(codes).sum().item()
            print(f">> Inf count: {inf_count}")

        # 检查codes的数值范围（normalized）
        print(f">> Codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")

        # 如果提供了code_stats，进行unnormalization
        if code_stats is not None:
            print(">> Unnormalizing codes...")
            codes = unnormalize_code(codes, code_stats)
            print(f">> Unnormalized codes min: {codes.min().item():.6f}, max: {codes.max().item():.6f}")
        else:
            print(">> No code_stats provided, returning normalized codes")

        # 清理内存
        del codes_all
        torch.cuda.empty_cache()

        return codes
