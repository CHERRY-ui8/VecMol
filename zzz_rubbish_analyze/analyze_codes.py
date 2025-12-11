import torch
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import lmdb
import pickle
import random
import traceback

# 自动找到项目根目录（包含funcmol目录的目录）
script_dir = Path(__file__).parent.absolute()
# 从脚本目录向上查找，直到找到包含funcmol目录的目录
project_root = script_dir
while project_root.parent != project_root:  # 直到根目录
    if (project_root / "funcmol").exists() and (project_root / "funcmol" / "configs").exists():
        break
    project_root = project_root.parent
else:
    # 如果找不到，尝试使用硬编码路径
    project_root = Path("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield")
    if not (project_root / "funcmol" / "configs").exists():
        project_root = Path("/home/huayuchen/Neurl-voxel")

# 添加项目根目录到Python路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import hydra
from funcmol.utils.utils_fm import add_noise_to_code
from funcmol.utils.utils_nf import normalize_code
import numpy as np

# 设置为cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def load_codes_from_lmdb(lmdb_path: str, keys_path: str, max_samples: int = None, random_sample: bool = False):
    """
    从LMDB数据库加载codes（支持采样，避免内存溢出）
    
    Args:
        lmdb_path: LMDB数据库路径
        keys_path: keys文件路径
        max_samples: 最大采样数量
        random_sample: 是否随机采样
    
    Returns:
        raw_codes: 采样后的codes
    """
    # 加载keys
    keys = torch.load(keys_path, weights_only=False)
    total_samples = len(keys)
    print(f"LMDB数据库包含 {total_samples} 个样本")
    
    # 确定采样数量
    if max_samples is None:
        # 不限制，使用全部样本
        indices = list(range(total_samples))
        print(f"使用全部 {total_samples} 个样本")
    else:
        sample_size = min(max_samples, total_samples)
        if total_samples > max_samples:
            if random_sample:
                # 随机采样
                indices = random.sample(range(total_samples), sample_size)
                indices.sort()  # 排序以便顺序读取
                print(f"随机采样 {sample_size} 个样本进行分析")
            else:
                # 顺序采样前N个
                indices = list(range(sample_size))
                print(f"顺序采样前 {sample_size} 个样本进行分析")
        else:
            indices = list(range(total_samples))
            print(f"使用全部 {total_samples} 个样本（未超过max_samples限制）")
    
    # 打开LMDB数据库
    db = lmdb.open(
        lmdb_path,
        map_size=10*(1024*1024*1024),  # 10GB
        create=False,
        subdir=True,
        readonly=True,
        lock=False,
        readahead=True,
        meminit=False,
        max_readers=1,
    )
    
    # 逐个加载样本
    all_codes = []
    print(f"正在从LMDB加载 {len(indices)} 个样本...")
    for i, idx in enumerate(indices):
        if (i + 1) % 1000 == 0:
            print(f"  已加载 {i + 1}/{len(indices)} 个样本...")
        
        key = keys[idx]
        # 确保key是bytes格式
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        # 从LMDB读取
        with db.begin(buffers=True) as txn:
            value = txn.get(key)
            if value is None:
                print(f"警告: 键 {key} 未找到，跳过")
                continue
            code_raw = pickle.loads(value)
            all_codes.append(code_raw)
    
    db.close()
    
    # 合并所有codes
    if not all_codes:
        print("错误: 未能加载任何codes")
        return None
    
    raw_codes = torch.stack(all_codes, dim=0)
    print(f"成功加载 {raw_codes.shape[0]} 个样本，shape: {raw_codes.shape}")
    return raw_codes


def load_codes(mode: str, codes_path: str = None, max_samples: int = None, random_sample: bool = False):
    """
    加载codes（真实或生成的）
    
    Args:
        mode: 'real' 或 'generated'
        codes_path: codes路径（real模式时可以是文件路径，generated模式时是目录路径）
    
    Returns:
        raw_codes: 原始codes
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mode == 'real':
        # 加载预计算的真实codes
        print(f"\n=== 加载预计算的Codes ===")
        if codes_path is None:
            codes_path = "/home/huayuchen/Neurl-voxel/exps/neural_field/nf_qm9/20250911/lightning_logs/version_1/checkpoints/codes/train/codes.pt"
        
        if not os.path.exists(codes_path):
            print(f"错误: codes路径不存在: {codes_path}")
            return None
        
        try:
            # 检查是文件还是目录
            if os.path.isdir(codes_path):
                # 如果是目录，优先检查LMDB
                lmdb_path = os.path.join(codes_path, "codes.lmdb")
                keys_path = os.path.join(codes_path, "codes_keys.pt")
                
                if os.path.exists(lmdb_path) and os.path.exists(keys_path):
                    # 使用LMDB加载
                    print(f"检测到LMDB格式，使用LMDB加载（支持大量数据）")
                    return load_codes_from_lmdb(lmdb_path, keys_path, max_samples, random_sample)
                
                # 如果不是LMDB，查找所有codes_*.pt文件
                codes_dir = Path(codes_path)
                code_files = sorted(list(codes_dir.glob("codes_*.pt")))
                if not code_files:
                    # 也尝试查找codes.pt
                    single_file = codes_dir / "codes.pt"
                    if single_file.exists():
                        code_files = [single_file]
                    else:
                        print(f"错误: 在目录 {codes_path} 中未找到任何codes文件 (codes_*.pt, codes.pt, 或 codes.lmdb)")
                        return None
                
                print(f"找到 {len(code_files)} 个codes文件:")
                for i, file in enumerate(code_files[:10]):  # 显示前10个
                    print(f"  {i+1}. {file.name}")
                if len(code_files) > 10:
                    print(f"  ... 还有 {len(code_files) - 10} 个文件")
                
                # 如果文件太多，建议使用LMDB
                if len(code_files) > 5:
                    print(f"⚠️  警告: 发现 {len(code_files)} 个codes文件，建议转换为LMDB格式以避免内存问题")
                    print(f"   转换命令: python funcmol/dataset/convert_codes_to_lmdb.py --codes_dir {os.path.dirname(codes_path)} --splits {os.path.basename(codes_path)}")
                
                # 加载所有codes文件
                all_codes = []
                for code_file in code_files:
                    codes = torch.load(code_file, map_location=device)
                    all_codes.append(codes)
                    print(f"  加载: {code_file.name} - shape: {codes.shape}")
                
                # 合并所有codes
                raw_codes = torch.cat(all_codes, dim=0)
                print(f"合并后的codes shape: {raw_codes.shape}")
            else:
                # 如果是单个文件，直接加载
                raw_codes = torch.load(codes_path, map_location=device)
                print(f"成功加载预计算的codes: {raw_codes.shape}")
            
            # 只取前max_samples个样本进行分析（如果指定了max_samples且样本太多）
            if max_samples is not None and raw_codes.shape[0] > max_samples:
                if random_sample:
                    # 随机采样
                    indices = random.sample(range(raw_codes.shape[0]), max_samples)
                    raw_codes = raw_codes[indices]
                    print(f"⚠️  样本数较多 ({raw_codes.shape[0]}), 随机采样 {max_samples} 个样本进行分析")
                else:
                    raw_codes = raw_codes[:max_samples]
                    print(f"⚠️  样本数较多 ({raw_codes.shape[0]}), 使用前 {max_samples} 个样本进行分析")
            
            return raw_codes
        except Exception as e:
            print(f"加载codes失败: {e}")
            traceback.print_exc()
            return None
    
    elif mode == 'generated':
        # 加载生成的codes
        print(f"\n=== 加载生成的Codes ===")
        codes_dir = Path(codes_path) if codes_path else None
        if codes_dir is None or not codes_dir.exists():
            print(f"错误: 目录不存在: {codes_dir}")
            return None
        
        # 查找所有code_*.pt文件
        code_files = list(codes_dir.glob("code_*.pt"))
        if not code_files:
            print(f"错误: 在目录 {codes_dir} 中未找到任何 code_*.pt 文件")
            return None
        
        print(f"\n=== 找到 {len(code_files)} 个codes文件 ===")
        for i, file in enumerate(code_files[:5]):  # 只显示前5个
            print(f"  {i+1}. {file.name}")
        if len(code_files) > 5:
            print(f"  ... 还有 {len(code_files) - 5} 个文件")
        
        # 加载所有codes
        all_codes = []
        for code_file in code_files:
            try:
                codes = torch.load(code_file, map_location='cpu')
                all_codes.append(codes)
                print(f"成功加载: {code_file.name} - shape: {codes.shape}")
            except Exception as e:
                print(f"加载失败: {code_file.name} - 错误: {e}")
        
        if not all_codes:
            print("错误: 没有成功加载任何codes")
            return None
        
        # 合并所有codes
        raw_codes = torch.cat(all_codes, dim=0)
        print(f"\n合并后的codes shape: {raw_codes.shape}")
        return raw_codes
    
    else:
        print(f"错误: 未知的mode: {mode}")
        return None

def analyze_codes(mode: str = 'real', codes_path: str = None, 
                  output_prefix: str = None, max_samples: int = None, 
                  random_sample: bool = False):
    """
    分析codes（真实或生成的）
    
    Args:
        mode: 'real' 或 'generated'
        codes_path: codes路径
        nf_checkpoint_path: neural field checkpoint路径（real模式可能需要）
        output_prefix: 输出文件前缀
    
    Returns:
        tuple: (raw_codes, normalized_codes, smooth_codes)
    """
    mode_name = "真实" if mode == 'real' else "生成"
    print(f"=== 分析{mode_name}的Codes ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载配置
    # 使用绝对路径指向配置文件目录
    config_dir = project_root / "funcmol" / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"配置文件目录不存在: {config_dir}")
    
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = hydra.compose(config_name="train_fm_qm9")
    
    print(f"\n=== 配置信息 ===")
    print(f"smooth_sigma: {cfg.smooth_sigma}")
    print(f"normalize_codes: {cfg.normalize_codes}")
    print(f"code_dim: {cfg.decoder.code_dim}")
    print(f"grid_size: {cfg.dset.grid_size}")
    
    # 2. 加载codes
    raw_codes = load_codes(mode, codes_path, max_samples=max_samples, random_sample=random_sample)
    if raw_codes is None:
        return None, None, None
    
    # 3. 分析codes
    print(f"\n=== 分析{mode_name}的Codes ===")
    
    # 对于generated模式，检查codes是否已经unnormalized
    is_normalized = None  # 初始化，只在generated模式时使用
    if mode == 'generated':
        codes_min = raw_codes.min().item()
        codes_max = raw_codes.max().item()
        codes_mean = raw_codes.mean().item()
        codes_std = raw_codes.std().item()
        
        print(f"生成的codes统计:")
        print(f"  最小值: {codes_min:.6f}")
        print(f"  最大值: {codes_max:.6f}")
        print(f"  均值: {codes_mean:.6f}")
        print(f"  标准差: {codes_std:.6f}")
        
        # 判断codes是否已经unnormalized
        if abs(codes_mean) < 0.1 and abs(codes_std - 1.0) < 0.1:
            print("  → 检测到codes可能是normalized的")
            is_normalized = True
        else:
            print("  → 检测到codes可能是unnormalized的（已还原）")
            is_normalized = False
    
    # 4. 分析raw codes
    raw_name = "Raw Codes" if mode == 'real' else "Unnormalized Codes"
    print(f"\n=== {raw_name} 分析 ===")
    analyze_codes_stats(raw_codes, raw_name)
    
    # 5. 处理normalized codes
    if mode == 'real':
        # 对于真实codes，总是进行normalization
        if cfg.normalize_codes:
            code_stats = {
                'mean': raw_codes.mean(dim=(0, 1), keepdim=True),
                'std': raw_codes.std(dim=(0, 1), keepdim=True)
            }
            normalized_codes = normalize_code(raw_codes, code_stats)
        else:
            normalized_codes = raw_codes
    else:
        # 对于生成codes，根据检测结果决定
        if is_normalized is not None and not is_normalized and cfg.normalize_codes:
            print(f"\n=== Normalize生成的Codes ===")
            code_stats = {
                'mean': raw_codes.mean(dim=(0, 1), keepdim=True),
                'std': raw_codes.std(dim=(0, 1), keepdim=True)
            }
            normalized_codes = normalize_code(raw_codes, code_stats)
        elif is_normalized is not None and is_normalized:
            normalized_codes = raw_codes
        else:
            normalized_codes = None
    
    if normalized_codes is not None:
        norm_name = "Normalized Codes" if mode == 'real' else "Denoised Codes"
        print(f"\n=== {norm_name} 分析 ===")
        analyze_codes_stats(normalized_codes, norm_name)
    
    # 6. （可选）应用denoiser加噪
    if mode == 'generated':
        # 生成的codes不再二次加噪
        smooth_codes = None
    else:
        codes_for_smooth = normalized_codes if normalized_codes is not None else raw_codes
        smooth_codes = add_noise_to_code(codes_for_smooth, smooth_sigma=cfg.smooth_sigma)
        smooth_name = "Smooth Codes" if mode == 'real' else "Smooth Generated Codes"
        print(f"\n=== {smooth_name} (Denoiser加噪后) 分析 ===")
        analyze_codes_stats(smooth_codes, smooth_name)
    
    # 7. 打印统计信息
    print(f"\n=== {mode_name} Codes统计信息 ===")
    print(f"Raw codes shape: {raw_codes.shape}")
    if normalized_codes is not None:
        print(f"Normalized codes shape: {normalized_codes.shape}")
    if smooth_codes is not None:
        print(f"Smooth codes shape: {smooth_codes.shape}")
    
    # 8. 创建可视化
    create_visualization(raw_codes, normalized_codes, smooth_codes, 
                        cfg.smooth_sigma, mode, output_prefix)
    
    # 9. 可选：real codes 分布诊断
    if mode == 'real' and globals().get('_DIAGNOSE_REAL', False):
        try:
            diagnose_real_codes(raw_codes, normalized_codes, output_prefix, n_show_channels=globals().get('_DIAG_N', 32))
        except Exception as e:
            print(f"诊断流程出错: {e}")

    # 10. 可选：体素维度诊断
    if mode == 'real' and globals().get('_VOXEL_DIAGNOSE', False):
        try:
            diagnose_voxel_codes(
                raw_codes,
                grid_size=cfg.dset.grid_size,
                output_prefix=output_prefix,
                reduce_method=globals().get('_VOXEL_REDUCE', 'mean'),
                topk_bc=globals().get('_VOXEL_TOPK', 32)
            )
        except Exception as e:
            print(f"体素诊断出错: {e}")

    return raw_codes, normalized_codes, smooth_codes


def _compute_channel_stats(x):
    """
    计算逐通道统计量：均值、方差、偏度、峰度、3σ越界比例、双峰系数BC。
    x: torch.Tensor, shape [num_samples, grid, code_dim] 或 [num_items, code_dim]
    返回: dict[str, np.ndarray] with length=code_dim
    """
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])  # [N, C]
    # 转CPU
    x = x.detach().cpu()
    N = x.shape[0]
    eps = 1e-8
    mean = x.mean(dim=0)
    std = x.std(dim=0) + eps
    z = (x - mean) / std
    # 偏度、峰度（Pearson）
    m3 = (z**3).mean(dim=0)
    skew = m3
    m4 = (z**4).mean(dim=0)
    kurtosis = m4  # Pearson kurtosis, 正态为3
    # 3σ越界
    exceed_3sigma = (z.abs() > 3).float().mean(dim=0)
    # 双峰系数（Bimodality Coefficient, 使用 Pearson kurtosis）
    # BC = (skew^2 + 1) / kurtosis，经验阈值 > 5/9 (~0.555) 往往提示双峰/多峰
    bc = (skew**2 + 1.0) / (kurtosis + eps)
    return {
        'mean': mean.numpy(),
        'std': (std - eps).numpy(),
        'skew': skew.numpy(),
        'kurtosis': kurtosis.numpy(),
        'exceed_3sigma': exceed_3sigma.numpy(),
        'bc': bc.numpy(),
    }


def diagnose_real_codes(raw_codes, normalized_codes=None, output_prefix=None, n_show_channels=32):
    """对 real codes 做统计诊断，保存 CSV 与汇总可视化。"""
    # 统计
    stats_raw = _compute_channel_stats(raw_codes)
    if normalized_codes is not None:
        stats_norm = _compute_channel_stats(normalized_codes)
    else:
        stats_norm = None

    code_dim = raw_codes.shape[-1]
    channel_ids = np.arange(code_dim)
    # 写 CSV
    header = ['channel','mean','std','skew','kurtosis','exceed_3sigma','bc']
    rows = []
    for c in range(code_dim):
        rows.append([
            c,
            float(stats_raw['mean'][c]), float(stats_raw['std'][c]),
            float(stats_raw['skew'][c]), float(stats_raw['kurtosis'][c]),
            float(stats_raw['exceed_3sigma'][c]), float(stats_raw['bc'][c])
        ])
    csv_name = 'codes_channel_stats.csv' if output_prefix is None else f'{output_prefix}_codes_channel_stats.csv'
    csv_path = f'/home/huayuchen/Neurl-voxel/{csv_name}'
    try:
        with open(csv_path, 'w') as f:
            f.write(','.join(header) + '\n')
            for r in rows:
                f.write(','.join(map(str, r)) + '\n')
        print(f"通道统计已保存到: {csv_name}")
    except Exception as e:
        print(f"写入CSV失败: {e}")

    # 可视化：代表性通道直方图 + 统计分布 + raw/normalized 对比
    # 选 n_show_channels 个通道（按BC排序取前半与后半各一半）
    order = np.argsort(stats_raw['bc'])
    half = max(1, n_show_channels // 2)
    pick = np.unique(np.concatenate([order[:half], order[-half:]]))

    # 准备数组
    data = raw_codes.detach().cpu().numpy().reshape(-1, code_dim)
    if normalized_codes is not None:
        data_norm = normalized_codes.detach().cpu().numpy().reshape(-1, code_dim)

    # 画图
    import matplotlib.pyplot as plt
    import math
    rows_plots = math.ceil(len(pick) / 4)
    cols_plots = 4
    fig = plt.figure(figsize=(4*cols_plots, 3*rows_plots + 6))

    # 第一部分：代表性通道直方图
    for idx, ch in enumerate(pick):
        ax = plt.subplot(rows_plots, cols_plots, idx+1)
        ax.hist(data[:, ch], bins=80, density=True, alpha=0.7, color='steelblue')
        m = data[:, ch].mean(); s = data[:, ch].std()
        ax.axvline(m-3*s, color='red', linestyle='--', label='-3σ')
        ax.axvline(m+3*s, color='red', linestyle='--', label='+3σ')
        ax.set_title(f'ch {int(ch)}  BC={stats_raw["bc"][ch]:.3f}')
        if idx == 0:
            ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.25, 1, 1])

    # 第二部分：统计量分布（附加在下方）
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, bottom=0.02, top=0.22, left=0.05, right=0.98, hspace=0.6, wspace=0.35)
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[0,1]); ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,0]); ax5 = fig.add_subplot(gs[1,1]); ax6 = fig.add_subplot(gs[1,2])

    ax1.hist(stats_raw['skew'], bins=40, color='tab:orange'); ax1.set_title('Skew (raw)')
    ax2.hist(stats_raw['kurtosis'], bins=40, color='tab:green'); ax2.set_title('Kurtosis (raw)')
    ax3.hist(stats_raw['bc'], bins=40, color='tab:purple'); ax3.set_title('Bimodality Coef (raw)')
    ax4.hist(stats_raw['exceed_3sigma'], bins=40, color='tab:red'); ax4.set_title('Exceed 3σ (raw)')
    ax5.hist(stats_raw['std'], bins=40, color='tab:blue'); ax5.set_title('Std (raw)')
    ax6.hist(stats_raw['mean'], bins=40, color='tab:gray'); ax6.set_title('Mean (raw)')
    for a in [ax1,ax2,ax3,ax4,ax5,ax6]:
        a.grid(True, alpha=0.3)

    # 第三部分：raw vs normalized 对比（若提供）
    if normalized_codes is not None:
        fig2, axes = plt.subplots(1, 2, figsize=(14,4))
        axes[0].hist(data.flatten(), bins=120, density=True, alpha=0.7, label='Raw', color='blue')
        axes[0].set_title('Raw Codes - All Channels')
        axes[0].grid(True, alpha=0.3)
        axes[1].hist(data_norm.flatten(), bins=120, density=True, alpha=0.7, label='Normalized', color='green')
        axes[1].set_title('Normalized Codes - All Channels')
        axes[1].grid(True, alpha=0.3)
        for a in axes:
            a.set_xlabel('Value'); a.set_ylabel('Density')

    # 保存
    diag_name = 'codes_diagnose.png' if output_prefix is None else f'{output_prefix}_codes_diagnose.png'
    path_plot = f'/home/huayuchen/Neurl-voxel/{diag_name}'
    plt.savefig(path_plot, dpi=300, bbox_inches='tight')
    print(f"诊断图已保存到: {diag_name}")

def _reduce_codes_to_scalar_series(codes, method='mean'):
    """
    将 [N, G, C] 的 codes 在通道维上聚合为标量序列 [N, G]。
    method: 'mean' 或 'pc1'（按每个体素做PC1投影）。
    """
    if method == 'mean':
        return codes.mean(dim=-1)
    elif method == 'pc1':
        with torch.no_grad():
            N, G, C = codes.shape
            x = codes.detach().cpu().numpy()
            s = np.zeros((N, G), dtype=np.float32)
            for g in range(G):
                X = x[:, g, :]
                Xc = X - X.mean(axis=0, keepdims=True)
                try:
                    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                    v = Vt[0]
                    s[:, g] = Xc @ v
                except Exception:
                    s[:, g] = X.mean(axis=1)
            return torch.from_numpy(s)
    else:
        raise ValueError(f"Unknown reduce method: {method}")

def _compute_voxel_stats(scalar_series, grid_size):
    """
    对每个体素统计 scalar_series 的分布特征。
    scalar_series: [N, G]
    返回: dict of 3D numpy arrays, 形状 [grid,grid,grid]
    """
    N, G = scalar_series.shape
    gs = grid_size
    x = scalar_series.detach().cpu().numpy()
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    z = (x - mean) / std
    skew = (z**3).mean(axis=0)
    kurt = (z**4).mean(axis=0)
    exceed = (np.abs(z) > 3).mean(axis=0)
    bc = (skew**2 + 1.0) / (kurt + 1e-8)
    def to3(v):
        return v.reshape(gs, gs, gs)
    return {
        'mean': to3(mean),
        'std': to3(std - 1e-8),
        'skew': to3(skew),
        'kurtosis': to3(kurt),
        'exceed_3sigma': to3(exceed),
        'bc': to3(bc),
    }

def diagnose_voxel_codes(raw_codes, grid_size, output_prefix=None, reduce_method='mean', topk_bc=32):
    """按体素维度诊断分布，输出CSV与二维投影热力图，以及BC最高体素的直方图。"""
    scalar_series = _reduce_codes_to_scalar_series(raw_codes, reduce_method)
    stats = _compute_voxel_stats(scalar_series, grid_size)

    gs = grid_size
    ii, jj, kk = np.meshgrid(np.arange(gs), np.arange(gs), np.arange(gs), indexing='ij')
    flat = {k: v.reshape(-1) for k, v in stats.items()}
    csv_name = 'voxel_stats.csv' if output_prefix is None else f'{output_prefix}_voxel_stats.csv'
    csv_path = f'/home/huayuchen/Neurl-voxel/{csv_name}'
    try:
        with open(csv_path, 'w') as f:
            f.write('index,i,j,k,mean,std,skew,kurtosis,exceed_3sigma,bc\n')
            for t in range(gs**3):
                f.write(','.join(map(str, [
                    t, int(ii.reshape(-1)[t]), int(jj.reshape(-1)[t]), int(kk.reshape(-1)[t]),
                    float(flat['mean'][t]), float(flat['std'][t]), float(flat['skew'][t]), float(flat['kurtosis'][t]),
                    float(flat['exceed_3sigma'][t]), float(flat['bc'][t])
                ])) + '\n')
        print(f"体素统计已保存到: {csv_name}")
    except Exception as e:
        print(f"写入体素CSV失败: {e}")

    import matplotlib.pyplot as plt
    proj = {k: v.max(axis=2) for k, v in stats.items()}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    keys = ['mean','std','skew','kurtosis','exceed_3sigma','bc']
    cmaps = ['viridis','magma','coolwarm','plasma','inferno','cividis']
    for ax, key, cmap in zip(axes.reshape(-1), keys, cmaps):
        im = ax.imshow(proj[key], origin='lower', cmap=cmap)
        ax.set_title(f'{key} (max-proj z)')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    name_proj = 'voxel_stats_projection.png' if output_prefix is None else f'{output_prefix}_voxel_stats_projection.png'
    path_proj = f'/home/huayuchen/Neurl-voxel/{name_proj}'
    plt.savefig(path_proj, dpi=300, bbox_inches='tight')
    print(f"体素统计投影图已保存到: {name_proj}")

    bc_flat = flat['bc']
    topk = min(topk_bc, gs**3)
    top_idx = np.argsort(bc_flat)[-topk:][::-1]
    vals = scalar_series.detach().cpu().numpy()
    import math
    rows = math.ceil(topk / 4)
    fig2, axes2 = plt.subplots(rows, 4, figsize=(16, 3*rows))
    axes2 = np.atleast_2d(axes2)
    for k in range(topk):
        g = top_idx[k]
        r, c = divmod(k, 4)
        ax = axes2[r, c]
        v = vals[:, g]
        m, s = v.mean(), v.std()
        ax.hist(v, bins=60, density=True, alpha=0.7, color='tab:purple')
        ax.axvline(m-3*s, color='red', linestyle='--'); ax.axvline(m+3*s, color='red', linestyle='--')
        ax.set_title(f'voxel {g} (bc={bc_flat[g]:.3f})')
        ax.grid(True, alpha=0.3)
    for k in range(topk, rows*4):
        r, c = divmod(k, 4)
        axes2[r, c].axis('off')
    plt.tight_layout()
    name_hist = 'voxel_topk_hist.png' if output_prefix is None else f'{output_prefix}_voxel_topk_hist.png'
    path_hist = f'/home/huayuchen/Neurl-voxel/{name_hist}'
    plt.savefig(path_hist, dpi=300, bbox_inches='tight')
    print(f"体素Top-BC直方图已保存到: {name_hist}")

def analyze_codes_stats(codes, name):
    """
    分析codes的统计信息（优化内存使用）
    """
    
    print(f"\n{name} 基本统计:")
    print(f"  形状: {codes.shape}")
    print(f"  总元素数: {codes.numel():,}")
    
    # 对于大数据集，使用更节省内存的方式计算
    total_elements = codes.numel()
    is_large = total_elements > 100_000_000  # 超过1亿元素
    
    if is_large:
        print(f"  ⚠️  数据量较大，使用分批计算以节省内存...")
        # 分批计算统计量
        codes_flat = codes.flatten()
        batch_size = 10_000_000  # 每批1000万元素
        
        # 基本统计（可以一次性计算，因为只是标量）
        min_val = codes.min().item()
        max_val = codes.max().item()
        mean_val = codes.mean().item()
        std_val = codes.std().item()
        
        # 中位数需要更多内存，对于大数据集可以跳过或采样
        try:
            median_val = codes.median().item()
        except RuntimeError:
            print(f"  ⚠️  中位数计算内存不足，跳过")
            median_val = None
        
        # 检查异常值
        nan_count = torch.isnan(codes).sum().item()
        inf_count = torch.isinf(codes).sum().item()
        
        print(f"  最小值: {min_val:.6f}")
        print(f"  最大值: {max_val:.6f}")
        print(f"  均值: {mean_val:.6f}")
        print(f"  标准差: {std_val:.6f}")
        if median_val is not None:
            print(f"  中位数: {median_val:.6f}")
        print(f"  NaN数量: {nan_count}")
        print(f"  Inf数量: {inf_count}")
        
        # 分批计算3σ异常值
        print(f"  正在计算3σ异常值（分批处理）...")
        lower_3sigma = mean_val - 3 * std_val
        upper_3sigma = mean_val + 3 * std_val
        
        beyond_3sigma_count = 0
        num_batches = (len(codes_flat) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(codes_flat))
            batch = codes_flat[start_idx:end_idx]
            beyond_3sigma_count += ((batch < lower_3sigma) | (batch > upper_3sigma)).sum().item()
            if (i + 1) % 10 == 0:
                print(f"    已处理 {i + 1}/{num_batches} 批次...")
        
        beyond_3sigma_ratio = beyond_3sigma_count / total_elements
        
        print(f"\n{name} 3σ异常值分析:")
        print(f"  3σ范围: [{lower_3sigma:.6f}, {upper_3sigma:.6f}]")
        print(f"  超出3σ的值数量: {beyond_3sigma_count:,} / {total_elements:,}")
        print(f"  超出3σ的值比例: {beyond_3sigma_ratio:.6f} ({beyond_3sigma_ratio*100:.4f}%)")
        print(f"  理论期望比例: 0.002700 (0.2700%)")
        print(f"  实际/理论比例: {beyond_3sigma_ratio/0.0027:.2f}x")
        
        # 判断状态
        if beyond_3sigma_ratio > 0.01:
            status = "⚠️ 异常"
        elif beyond_3sigma_ratio > 0.005:
            status = "⚠️ 注意"
        else:
            status = "✅ 正常"
        print(f"  状态: {status}")
        
        # 对于大数据集，只计算1σ和3σ，跳过其他
        print(f"\n{name} 不同σ范围异常值比例（大数据集仅计算关键值）:")
        for sigma in [1, 3]:
            lower_bound = mean_val - sigma * std_val
            upper_bound = mean_val + sigma * std_val
            
            beyond_count = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(codes_flat))
                batch = codes_flat[start_idx:end_idx]
                beyond_count += ((batch < lower_bound) | (batch > upper_bound)).sum().item()
            
            beyond_ratio = beyond_count / total_elements
            theoretical_ratios = {1: 0.3173, 3: 0.0027}
            theoretical_ratio = theoretical_ratios[sigma]
            print(f"  ±{sigma}σ: {beyond_ratio:.4f} ({beyond_ratio*100:5.2f}%) [理论: {theoretical_ratio:.4f} ({theoretical_ratio*100:5.2f}%)]")
        
        # 释放内存
        del codes_flat
        torch.cuda.empty_cache() if codes.is_cuda else None
        
    else:
        # 小数据集，使用原有方法
        print(f"  最小值: {codes.min().item():.6f}")
        print(f"  最大值: {codes.max().item():.6f}")
        print(f"  均值: {codes.mean().item():.6f}")
        print(f"  标准差: {codes.std().item():.6f}")
        print(f"  中位数: {codes.median().item():.6f}")
        
        # 检查异常值
        nan_count = torch.isnan(codes).sum().item()
        inf_count = torch.isinf(codes).sum().item()
        print(f"  NaN数量: {nan_count}")
        print(f"  Inf数量: {inf_count}")
        
        # 分析3σ异常值比例
        mean_val = codes.mean().item()
        std_val = codes.std().item()
        lower_3sigma = mean_val - 3 * std_val
        upper_3sigma = mean_val + 3 * std_val
        
        beyond_3sigma = ((codes < lower_3sigma) | (codes > upper_3sigma)).float()
        beyond_3sigma_ratio = beyond_3sigma.mean().item()
        beyond_3sigma_count = beyond_3sigma.sum().item()
        
        print(f"\n{name} 3σ异常值分析:")
        print(f"  3σ范围: [{lower_3sigma:.6f}, {upper_3sigma:.6f}]")
        print(f"  超出3σ的值数量: {beyond_3sigma_count:,} / {codes.numel():,}")
        print(f"  超出3σ的值比例: {beyond_3sigma_ratio:.6f} ({beyond_3sigma_ratio*100:.4f}%)")
        print(f"  理论期望比例: 0.002700 (0.2700%)")
        print(f"  实际/理论比例: {beyond_3sigma_ratio/0.0027:.2f}x")
        
        # 判断状态
        if beyond_3sigma_ratio > 0.01:
            status = "⚠️ 异常"
        elif beyond_3sigma_ratio > 0.005:
            status = "⚠️ 注意"
        else:
            status = "✅ 正常"
        print(f"  状态: {status}")
        
        # 分析不同σ范围的异常值比例
        print(f"\n{name} 不同σ范围异常值比例:")
        for sigma in [1, 2, 3, 4, 5]:
            lower_bound = mean_val - sigma * std_val
            upper_bound = mean_val + sigma * std_val
            beyond_sigma = ((codes < lower_bound) | (codes > upper_bound)).float()
            beyond_ratio = beyond_sigma.mean().item()
            
            theoretical_ratios = [0.3173, 0.0455, 0.0027, 0.0001, 0.0000]
            theoretical_ratio = theoretical_ratios[sigma-1]
            
            print(f"  ±{sigma}σ: {beyond_ratio:.4f} ({beyond_ratio*100:5.2f}%) [理论: {theoretical_ratio:.4f} ({theoretical_ratio*100:5.2f}%)]")

def create_visualization(raw_codes, normalized_codes, smooth_codes, smooth_sigma, 
                        mode='real', output_prefix=None):
    """创建可视化图表"""
    
    # 设置颜色和标题
    smooth_title = ''
    if mode == 'real':
        raw_color = 'blue'
        raw_title = 'Raw Codes (Encoder Output)'
        norm_title = 'Normalized Codes'
        smooth_title = f'Smooth Codes (σ={smooth_sigma})'
        comparison_title = 'Codes Distribution Comparison'
        output_file = 'codes_analysis.png'
    else:
        raw_color = 'purple'
        raw_title = 'Unnormalized Codes'
        norm_title = 'Denoised Codes'
        comparison_title = 'Codes Distribution Comparison'
        output_file = 'generated_codes_analysis.png'
    
    if output_prefix:
        output_file = f"{output_prefix}_{output_file}"
    
    # 转换为numpy
    raw_np = raw_codes.detach().cpu().numpy().flatten()
    
    # 创建图表
    if mode == 'generated':
        # 生成模式下：展示 Denoised（normalized_codes）、Unnormalized（raw_codes）以及对比
        if normalized_codes is not None:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            norm_np = normalized_codes.detach().cpu().numpy().flatten()

            # 1. Denoised codes 分布
            axes[0, 0].hist(norm_np, bins=100, alpha=0.7, density=True, color='green')
            axes[0, 0].axvline(norm_np.mean() - 3*norm_np.std(), color='red', linestyle='--', label='-3σ')
            axes[0, 0].axvline(norm_np.mean() + 3*norm_np.std(), color='red', linestyle='--', label='+3σ')
            axes[0, 0].set_title(norm_title)
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Unnormalized codes 分布
            axes[0, 1].hist(raw_np, bins=100, alpha=0.7, density=True, color=raw_color)
            axes[0, 1].axvline(raw_np.mean() - 3*raw_np.std(), color='red', linestyle='--', label='-3σ')
            axes[0, 1].axvline(raw_np.mean() + 3*raw_np.std(), color='red', linestyle='--', label='+3σ')
            axes[0, 1].set_title(raw_title)
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Density')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. 对比分布
            axes[1, 1].hist(raw_np, bins=50, alpha=0.5, density=True, label=raw_title, color=raw_color)
            axes[1, 1].hist(norm_np, bins=50, alpha=0.5, density=True, label=norm_title, color='green')
            axes[1, 1].set_title(comparison_title)
            axes[1, 1].set_xlabel('Value')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            # 4. 占位说明
            axes[1, 0].axis('off')
            axes[1, 0].text(0.5, 0.5, 'No smoothing in generated mode', ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            # 只显示 Unnormalized，并提示缺少 Denoised
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            axes[0].hist(raw_np, bins=100, alpha=0.7, density=True, color=raw_color)
            axes[0].axvline(raw_np.mean() - 3*raw_np.std(), color='red', linestyle='--', label='-3σ')
            axes[0].axvline(raw_np.mean() + 3*raw_np.std(), color='red', linestyle='--', label='+3σ')
            axes[0].set_title(raw_title)
            axes[0].set_xlabel('Value')
            axes[0].set_ylabel('Density')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[1].text(0.5, 0.5, 'No denoised codes available', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('No Comparison Available')
    elif normalized_codes is not None and smooth_codes is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        norm_np = normalized_codes.detach().cpu().numpy().flatten()
        smooth_np = smooth_codes.detach().cpu().numpy().flatten()
        
        # 1. Raw codes分布
        axes[0, 0].hist(raw_np, bins=100, alpha=0.7, density=True, color=raw_color)
        axes[0, 0].axvline(raw_np.mean() - 3*raw_np.std(), color='red', linestyle='--', label='-3σ')
        axes[0, 0].axvline(raw_np.mean() + 3*raw_np.std(), color='red', linestyle='--', label='+3σ')
        axes[0, 0].set_title(raw_title)
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Normalized codes分布
        axes[0, 1].hist(norm_np, bins=100, alpha=0.7, density=True, color='green')
        axes[0, 1].axvline(norm_np.mean() - 3*norm_np.std(), color='red', linestyle='--', label='-3σ')
        axes[0, 1].axvline(norm_np.mean() + 3*norm_np.std(), color='red', linestyle='--', label='+3σ')
        axes[0, 1].set_title(norm_title)
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Smooth codes分布
        axes[1, 0].hist(smooth_np, bins=100, alpha=0.7, density=True, color='orange')
        axes[1, 0].axvline(smooth_np.mean() - 3*smooth_np.std(), color='red', linestyle='--', label='-3σ')
        axes[1, 0].axvline(smooth_np.mean() + 3*smooth_np.std(), color='red', linestyle='--', label='+3σ')
        axes[1, 0].set_title(smooth_title)
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 对比分布
        axes[1, 1].hist(raw_np, bins=50, alpha=0.5, density=True, label=raw_title.split('(')[0].strip(), color=raw_color)
        axes[1, 1].hist(norm_np, bins=50, alpha=0.5, density=True, label='Normalized Codes', color='green')
        axes[1, 1].hist(smooth_np, bins=50, alpha=0.5, density=True, label='Smooth Codes', color='orange')
        axes[1, 1].set_title(comparison_title)
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Raw codes分布
        axes[0].hist(raw_np, bins=100, alpha=0.7, density=True, color=raw_color)
        axes[0].axvline(raw_np.mean() - 3*raw_np.std(), color='red', linestyle='--', label='-3σ')
        axes[0].axvline(raw_np.mean() + 3*raw_np.std(), color='red', linestyle='--', label='+3σ')
        axes[0].set_title(raw_title)
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 如果有smooth codes，显示对比
        if smooth_codes is not None:
            smooth_np = smooth_codes.detach().cpu().numpy().flatten()
            axes[1].hist(raw_np, bins=50, alpha=0.5, density=True, label=raw_title.split('(')[0].strip(), color=raw_color)
            axes[1].hist(smooth_np, bins=50, alpha=0.5, density=True, label='Smooth Codes', color='orange')
            axes[1].set_title(f'{raw_title.split("(")[0].strip()} vs Smooth Codes')
            axes[1].set_xlabel('Value')
            axes[1].set_ylabel('Density')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No additional codes to compare', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('No Comparison Available')
    
    plt.tight_layout()
    output_path = f'/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/zzz_rubbish_analyze/{output_file}'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化图表已保存到: {output_file}")
    plt.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='分析codes（真实或生成的）')
    parser.add_argument('--mode', choices=['real', 'generated'], default='real',
                       help='分析模式: real (真实codes) 或 generated (生成codes)')
    parser.add_argument('--codes_path', type=str, 
                       help='codes路径: real模式为.pt文件路径，generated模式为包含code_*.pt的目录路径')
    parser.add_argument('--diagnose', action='store_true', help='real模式下输出每通道统计与诊断图')
    parser.add_argument('--diagnose_n_channels', type=int, default=32, help='诊断图中展示的代表通道数目')
    parser.add_argument('--voxel_diagnose', action='store_true', help='real模式下进行体素维度分布诊断')
    parser.add_argument('--voxel_reduce', choices=['mean','pc1'], default='mean', help='体素诊断的通道聚合方式')
    parser.add_argument('--voxel_topk_bc', type=int, default=32, help='体素诊断中按BC展示的Top-K直方图数量')
    parser.add_argument('--max_samples', type=int, default=None, help='最大采样数量（默认None表示加载全部数据，用于LMDB或大量数据时可限制采样）')
    parser.add_argument('--random_sample', action='store_true', default=False, help='是否随机采样（仅当指定--max_samples时有效）')
    
    args = parser.parse_args()
    
    try:
        # 将诊断开关暴露给上游流程
        _DIAGNOSE_REAL = bool(args.diagnose)
        _DIAG_N = int(args.diagnose_n_channels)
        _VOXEL_DIAGNOSE = bool(args.voxel_diagnose)
        _VOXEL_REDUCE = str(args.voxel_reduce)
        _VOXEL_TOPK = int(args.voxel_topk_bc)
        raw_codes, normalized_codes, smooth_codes = analyze_codes(
            mode=args.mode,
            codes_path=args.codes_path,
            max_samples=args.max_samples,
            random_sample=args.random_sample
        )
        
        if raw_codes is not None:
            print("\n分析完成！")
        else:
            print("分析失败！")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
