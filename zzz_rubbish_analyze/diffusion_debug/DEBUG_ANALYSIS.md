# Diffusion 训练调试分析

## 训练流程概览

```
main() 
  → FuncmolLightningModule._training_step_ddpm()
    → FuncMol.train_ddpm_step()
      → compute_ddpm_loss_x0() (在 ddpm.py)
        → q_sample() (前向扩散)
        → model.forward() (模型预测)
        → 损失计算
```

## 关键断点位置和需要记录的变量

### 1. 训练入口 - `train_fm_lt.py`

**位置1: `_training_step_ddpm()` 函数入口 (line 140)**
- **变量**: 
  - `codes`: 原始输入数据 [B, N*N*N, code_dim]
  - `codes.shape`, `codes.min()`, `codes.max()`, `codes.mean()`, `codes.std()`
  - `batch_idx`: 当前batch索引

**位置2: `_training_step_ddpm()` 函数调用 train_ddpm_step 前 (line 168)**
- **变量**:
  - `codes`: 确认输入数据正确
  - `self.funcmol.diffusion_method`: 确认使用的方法

### 2. DDPM损失计算 - `ddpm.py`

**位置3: `compute_ddpm_loss_x0()` 函数入口 (line 398)**
- **变量**:
  - `x_0`: 原始数据 [B, N*N*N, code_dim]
  - `x_0.shape`, `x_0.min()`, `x_0.max()`, `x_0.mean()`, `x_0.std()`
  - `num_timesteps`: 总时间步数
  - `use_time_weight`: 是否使用时间权重

**位置4: 时间步采样后 (line 417)**
- **变量**:
  - `t`: 随机采样的时间步 [B]
  - `t.min()`, `t.max()`, `t.mean()`: 时间步分布
  - `t` 的直方图（可选，记录前几个值）

**位置5: 噪声生成后 (line 420)**
- **变量**:
  - `noise`: 生成的噪声 [B, N*N*N, code_dim]
  - `noise.shape`, `noise.min()`, `noise.max()`, `noise.mean()`, `noise.std()`

**位置6: 前向扩散后 (line 423)**
- **变量**:
  - `x_t`: 加噪后的数据 [B, N*N*N, code_dim]
  - `x_t.shape`, `x_t.min()`, `x_t.max()`, `x_t.mean()`, `x_t.std()`
  - 验证: `x_t` 应该介于 `x_0` 和 `noise` 之间

**位置7: 模型预测后 (line 426)**
- **变量**:
  - `predicted_x0`: 模型预测的 x0 [B, N*N*N, code_dim]
  - `predicted_x0.shape`, `predicted_x0.min()`, `predicted_x0.max()`, `predicted_x0.mean()`, `predicted_x0.std()`
  - 检查 NaN/Inf: `torch.isnan(predicted_x0).any()`, `torch.isinf(predicted_x0).any()`

**位置8: 损失计算后 (line 430-440)**
- **变量**:
  - `loss_per_sample`: 每个样本的损失 [B]
  - `loss_per_sample.min()`, `loss_per_sample.max()`, `loss_per_sample.mean()`
  - `weights`: 时间步权重（如果使用）[B]
  - `loss`: 最终损失值（标量）
  - 检查 NaN/Inf: `torch.isnan(loss)`, `torch.isinf(loss)`

### 3. 前向扩散过程 - `ddpm.py`

**位置9: `q_sample()` 函数内部 (line 98-121)**
- **变量**:
  - `sqrt_alphas_cumprod_t`: 提取的alpha累积乘积的平方根
  - `sqrt_one_minus_alphas_cumprod_t`: 提取的(1-alpha)累积乘积的平方根
  - 验证扩散公式: `x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise`

### 4. 模型前向传播 - `funcmol.py`

**位置10: `forward()` 函数入口 (line 95)**
- **变量**:
  - `y`: 输入张量 [B, N*N*N, code_dim]
  - `t`: 时间步 [B]
  - `y.shape`, `y.min()`, `y.max()`, `y.mean()`, `y.std()`
  - `t.min()`, `t.max()`, `t.mean()`

**位置11: `forward()` 函数输出 (line 115)**
- **变量**:
  - `xhat`: 模型输出 [B, N*N*N, code_dim]
  - `xhat.shape`, `xhat.min()`, `xhat.max()`, `xhat.mean()`, `xhat.std()`
  - 检查 NaN/Inf

### 5. 扩散常数 - `ddpm.py`

**位置12: `prepare_diffusion_constants()` 函数 (line 51-92)**
- **变量**:
  - `betas`: β序列
  - `alphas_cumprod`: α累积乘积
  - `sqrt_alphas_cumprod`: sqrt(α̅_t)
  - `sqrt_one_minus_alphas_cumprod`: sqrt(1-α̅_t)
  - 记录前10个和后10个时间步的值

## 调试策略

### 假设列表

1. **假设A**: 输入数据 `codes` 的数值范围异常（过大或过小）
2. **假设B**: 时间步 `t` 的采样分布不均匀
3. **假设C**: 前向扩散过程 `q_sample()` 计算错误
4. **假设D**: 模型预测 `predicted_x0` 输出异常（NaN/Inf或数值过大）
5. **假设E**: 损失计算过程中出现数值不稳定
6. **假设F**: 扩散常数 `diffusion_consts` 计算错误

### 日志记录频率

- **每个batch**: 记录位置1, 2, 3, 4, 5, 6, 7, 8
- **每10个batch**: 记录位置9, 10, 11（避免日志过多）
- **初始化时**: 记录位置12（扩散常数只需记录一次）

### 关键检查点

1. **数值范围检查**: 所有张量的 min/max/mean/std 应在合理范围内
2. **NaN/Inf检查**: 模型输出和损失值不应包含 NaN 或 Inf
3. **形状一致性**: 确保所有张量的形状符合预期
4. **设备一致性**: 确保所有张量在同一设备上（CPU/GPU）
5. **梯度检查**: 可选，检查梯度是否正常（不爆炸/消失）

## 实施建议

1. 使用 Python 的 `json` 库写入 NDJSON 格式日志
2. 每个日志条目包含：位置、变量名、变量值（统计信息）、时间戳
3. 使用 `hypothesisId` 字段关联到对应的假设
4. 在关键位置添加日志，但避免过度记录导致性能下降
5. 使用可折叠的代码区域包裹日志代码，保持代码整洁
