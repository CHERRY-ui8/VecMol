import matplotlib.pyplot as plt
import torch
import os
from funcmol.models.ddpm import get_beta_schedule, prepare_diffusion_constants

if __name__ == "__main__":
    num_timesteps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    schedules = ["linear", "cosine", "sigmoid"]  # 要比较的调度
    save_path = "beta_alpha_schedules.png"
    timesteps = torch.arange(1, num_timesteps + 1)

    plt.figure(figsize=(12, 5))

    # ---------- β_t 对比 ----------
    plt.subplot(1, 2, 1)
    for schedule in schedules:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule=schedule)
        plt.plot(timesteps, betas.numpy(), label=schedule)
    plt.title(r"Comparison of $\beta_t$ schedules")
    plt.xlabel("timestep")
    plt.ylabel(r"$\beta_t$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # ---------- \bar{α}_t 对比 ----------
    plt.subplot(1, 2, 2)
    for schedule in schedules:
        betas = get_beta_schedule(beta_start, beta_end, num_timesteps, schedule=schedule)
        diffusion_consts = prepare_diffusion_constants(betas)
        alphas_cumprod = diffusion_consts["alphas_cumprod"]
        plt.plot(timesteps, alphas_cumprod.numpy(), label=schedule)
    plt.title(r"Comparison of $\bar{\alpha}_t$ schedules")
    plt.xlabel("timestep")
    plt.ylabel(r"$\bar{\alpha}_t$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)   # ← 保存图片
    print(f"✅ 图像已保存到：{save_path}")
