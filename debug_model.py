import os
import sys
import torch
import hydra
from pathlib import Path
from lightning import Fabric

# 设置 torch.compile 兼容性
import torch._dynamo
torch._dynamo.config.suppress_errors = True

## set up environment
project_root = Path(os.getcwd()).parent
sys.path.insert(0, str(project_root))

from funcmol.gnf_visualizer import (
    setup_environment, load_config_from_exp_dir, load_model, 
    create_converter, prepare_data
)

# 模型根目录
model_root = "/home/huayuchen/funcmol-main-neuralfield/funcmol/exps/neural_field"

exp_name = 'nf_qm9_20250803_150822_593945'
sample_idx = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = os.path.join(model_root, exp_name)

# 从实验目录加载配置
config = load_config_from_exp_dir(model_dir)

# 准备数据
fabric = Fabric(
    accelerator="auto",
    devices=1,
    precision="32-true",
    strategy="auto"
)
fabric.launch()

batch, gt_coords, gt_types = prepare_data(fabric, config, device)

# 加载模型
encoder, decoder = load_model(fabric, config, model_dir=model_dir)

# 生成 codes
with torch.no_grad():
    codes = encoder(batch)

# 创建converter
converter = create_converter(config, device)

print(f"converter.n_query_points: {converter.n_query_points}")
print(f"codes shape: {codes.shape}")

# 模拟field_func调用
z = torch.randn(converter.n_query_points, 3, device=device)  # [3000, 3]

# 模拟predicted_field_func中的处理
if z.dim() == 2:  # [n_points, 3]
    z = z.unsqueeze(0)  # [1, n_points, 3]
elif z.dim() == 3:  # [batch, n_points, 3]
    pass
else:
    raise ValueError(f"Unexpected points shape: {z.shape}")

print(f"z after unsqueeze shape: {z.shape}")
print(f"codes[sample_idx:sample_idx+1] shape: {codes[sample_idx:sample_idx+1].shape}")

# 尝试调用decoder
try:
    result = decoder(z, codes[sample_idx:sample_idx+1])
    print(f"decoder result shape: {result.shape}")
    print("decoder调用成功")
except Exception as e:
    print(f"decoder调用失败: {e}")
    import traceback
    traceback.print_exc() 