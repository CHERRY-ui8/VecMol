# FuncMol 架构详细说明

## 1. 整体架构概述

FuncMol是一个基于神经场（Neural Field）的3D分子生成模型，采用两阶段训练策略：

1. **神经场自编码器（Neural Field Autoencoder）**：学习分子的连续场表示
2. **扩散生成模型（DDPM）**：在潜在码空间生成新的分子

## 2. 核心组件架构

### 2.1 编码器（CrossGraphEncoder）

**功能**：将3D分子结构编码为潜在码（latent codes）

**输入**：
- 原子坐标：`[N_atoms, 3]`
- 原子类型：`[N_atoms]`

**输出**：
- 潜在码：`[B, grid_size³, code_dim]`，其中`grid_size³`是锚点网格的总点数

**架构细节**：

1. **节点初始化**：
   - 原子节点：使用one-hot编码的原子类型，填充到`code_dim`维度
   - 网格节点：初始化为零向量

2. **图构建**：
   - **原子内部图**：使用k-NN（`atom_k_neighbors`）连接原子之间
   - **原子-网格交叉图**：使用k-NN（`k_neighbors`）连接每个网格锚点到最近的原子

3. **消息传递**：
   - 使用`MessagePassingGNN`层进行多层消息传递
   - 边特征：使用Gaussian Smearing对距离进行编码
   - 消息函数：`m_ij = MLP([h_i, h_j, dist_embedding])`
   - 聚合：使用mean aggregation
   - 残差连接：`h_new = h + LayerNorm(aggregated_messages)`

4. **输出**：提取网格节点的特征作为潜在码

**关键参数**：
- `grid_size`: 锚点网格大小（通常为8，得到8³=512个锚点）
- `code_dim`: 潜在码维度（通常为128或1024）
- `k_neighbors`: 网格到原子的连接数（通常为32）
- `atom_k_neighbors`: 原子之间的连接数（通常为8）
- `cutoff`: 距离截断阈值（通常为5.0 Å）

### 2.2 解码器（EGNNVectorField）

**功能**：从潜在码和查询点预测矢量场

**输入**：
- 查询点：`[B, n_points, 3]`
- 潜在码：`[B, grid_size³, code_dim]`

**输出**：
- 矢量场：`[B, n_points, n_atom_types, 3]`，每个查询点对每个原子类型预测一个3D向量

**架构细节**：

1. **节点初始化**：
   - 查询点节点：初始化为零特征
   - 网格锚点节点：使用潜在码作为初始特征

2. **图构建**：
   - 使用k-NN（`k_neighbors=32`）连接每个查询点到最近的网格锚点

3. **EGNN层（E(n) Equivariant Graph Neural Network）**：
   - **消息计算**：
     ```
     m_ij = EdgeMLP([h_i, h_j, ||x_i - x_j||])
     ```
   - **坐标更新**（等变）：
     ```
     coord_coef = CoordMLP(m_ij)  # 标量系数
     direction = (x_j - x_i) / ||x_j - x_i||  # 单位方向向量
     Δx_i = coord_coef * direction  # 标量乘以方向
     x_i_new = x_i + aggregate(Δx_i)  # 残差连接
     ```
   - **特征更新**：
     ```
     m_aggr = aggregate(m_ij)
     h_i_new = h_i + NodeMLP([h_i, m_aggr])  # 残差连接
     ```
   - **Cutoff函数**：使用余弦截断函数平滑衰减远距离连接

4. **场预测层**：
   - 最后一层EGNN预测`out_x_dim = n_atom_types`个坐标更新
   - 计算相对于查询点的残差向量：
     ```
     vector_field = predicted_sources - query_points
     ```

**关键特性**：
- **E(n)等变性**：对旋转和平移等变，保证几何一致性
- **多原子类型支持**：为每个原子类型独立预测矢量场
- **空间局部性**：通过k-NN和cutoff保证局部感受野

### 2.3 去噪器（GNNDenoiser）

**功能**：在潜在码空间进行去噪，用于DDPM生成

**输入**：
- 带噪声的潜在码：`[B, grid_size³, code_dim]`
- 时间步：`[B]`（用于DDPM）

**输出**：
- 去噪后的潜在码：`[B, grid_size³, code_dim]`（或预测的噪声/原始数据）

**架构细节**：

1. **输入投影**：
   ```
   h = InputProjection(y)  # [B, grid_size³, hidden_dim]
   ```

2. **时间嵌入**（DDPM模式）：
   ```
   t_emb = SinusoidalEmbedding(t, time_emb_dim)
   t_emb_proj = TimeProjection(t_emb)  # [B, hidden_dim]
   t_emb_broadcast = expand(t_emb_proj)  # [B, grid_size³, hidden_dim]
   ```

3. **EGNN去噪层**：
   - 使用`EGNNDenoiserLayer`进行多层消息传递
   - 图构建：使用`radius_graph`在网格锚点之间构建图（`radius`参数）
   - 消息函数：`m_ij = EdgeMLP([h_i, h_j, dist, t_emb_i])`（包含时间嵌入）
   - 特征更新：`h_i_new = h_i + NodeMLP([h_i, m_aggr])`

4. **输出投影**：
   ```
   output = OutputProjection(h)  # [B, grid_size³, code_dim]
   ```

**两种模式**：
- **预测噪声（epsilon）**：`ε_θ(x_t, t)`
- **预测原始数据（x0）**：`x_0_θ(x_t, t)`

### 2.4 EGNN层详解

**E(n) Equivariant Graph Neural Network**是核心架构，保证几何等变性。

**数学形式**：

对于节点`i`和邻居`j`：

1. **消息计算**：
   ```
   m_ij = φ_e(h_i, h_j, ||x_i - x_j||)
   ```
   其中`φ_e`是边MLP，输入是节点特征和距离。

2. **坐标更新**（等变）：
   ```
   a_ij = φ_x(m_ij)  # 标量系数
   x_i ← x_i + (1/|N(i)|) Σ_{j∈N(i)} a_ij · (x_j - x_i) / ||x_j - x_i||
   ```
   这保证了旋转和平移等变性。

3. **特征更新**（不变）：
   ```
   m_i = (1/|N(i)|) Σ_{j∈N(i)} m_ij
   h_i ← h_i + φ_h(h_i, m_i)
   ```

**关键优势**：
- 自动满足E(n)群等变性
- 无需显式的旋转矩阵
- 适合处理3D几何结构

### 2.5 场转换器（GNFConverter）

**功能**：在分子结构和矢量场之间转换

#### 2.5.1 分子到场（mol2gnf）

**输入**：
- 原子坐标：`[B, N_atoms, 3]`
- 原子类型：`[B, N_atoms]`
- 查询点：`[B, n_points, 3]`

**输出**：
- 矢量场：`[B, n_points, n_atom_types, 3]`

**场定义方法总览**：

在开发过程中，我们探索了多种场定义方法，最终通过QM9数据集上的重建效果对比，确定了最优方法。所有方法的核心思想是：为每个查询点计算指向原子的梯度向量，使得在梯度上升过程中，采样点能够收敛到原子位置。

**基础方法**：

1. **Gaussian Field**：
   ```
   dist_ij = ||query_i - atom_j||
   weight_j = exp(-dist_ij² / (2σ²)) / σ²
   vector_field[i, type_j] = weight_j · (atom_j - query_i)
   ```
   直接使用高斯核函数，权重随距离指数衰减。

2. **Softmax Field**：
   ```
   dist_ij = ||query_i - atom_j||
   weight_j = softmax(-dist_ij / temperature)
   vector_field[i, type_j] = weight_j · (atom_j - query_i) / ||atom_j - query_i||
   ```
   使用softmax归一化，确保权重和为1，倾向于选择最近的原子。

3. **Softmax Normalized (sfnorm)**：
   ```
   dist_ij = ||query_i - atom_j||
   weight_j = softmax(-dist_ij / temperature)
   direction = (atom_j - query_i) / ||atom_j - query_i||
   vector_field[i, type_j] = weight_j · direction
   ```
   在softmax基础上，显式归一化方向向量。

4. **LogSumExp**：
   ```
   dist_ij = ||query_i - atom_j||
   gaussian_grad = exp(-dist_ij² / (2σ²)) / σ² · (atom_j - query_i)
   magnitude = ||gaussian_grad||
   direction = gaussian_grad / magnitude
   log_sum_mag = logsumexp(magnitude)
   vector_field[i, type_j] = scale · direction · log_sum_mag
   ```
   使用log-sum-exp聚合梯度模长，保持数值稳定性。

5. **Inverse Square**：
   ```
   dist_ij = ||query_i - atom_j||
   weight_j = strength / (dist_ij² + ε)
   vector_field[i, type_j] = weight_j · (atom_j - query_i) / ||atom_j - query_i||
   ```
   使用距离平方反比，模拟物理场（如引力场）。

6. **Distance**：
   ```
   dist_ij = ||query_i - atom_j||
   w_softmax = softmax(-dist_ij / sig_sf)
   w_mag = clamp(dist_ij, 0, 1)
   direction = (atom_j - query_i) / ||atom_j - query_i||
   vector_field[i, type_j] = sum(w_softmax · w_mag · direction)
   ```
   使用距离值作为magnitude，简单但有效。

**复合方法（最终选择）**：

受到高斯、softmax等结构的启发，我们设计了**复合场定义方法**，将**方向选择**和**模长控制**分离，通过不同的组合实现更灵活和有效的场表示。核心思想是：

```
vector_field = Σ (w_softmax · w_mag · diff_normed)
```

其中：
- `w_softmax`: 控制方向选择，使用softmax选择最近的原子
- `w_mag`: 控制梯度模长（magnitude），影响收敛速度和稳定性
- `diff_normed`: 归一化的方向向量，指向原子

**关键参数**：
- `sig_sf` (softmax field sigma): 控制softmax的尖锐程度，影响方向选择的局部性
- `sig_mag` (magnitude sigma): 控制magnitude函数的形状，影响梯度模长的衰减特性

**7. Gaussian Magnitude (gaussian_mag)** - 决赛圈方法：

```
dist_ij = ||query_i - atom_j||
w_softmax = softmax(-dist_ij / sig_sf)  // 方向选择：选择最近的原子
w_mag = exp(-dist_ij² / (2·sig_mag²)) · dist_ij  // 高斯衰减 × 距离
diff_normed = (atom_j - query_i) / ||atom_j - query_i||  // 归一化方向
vector_field[i, type_j] = Σ(w_softmax · w_mag · diff_normed)
```

**设计思路**：
- `w_softmax`: 使用softmax确保权重归一化，倾向于选择最近的原子，提供稳定的方向指引
- `w_mag`: 结合高斯衰减（`exp(-dist²/(2σ²))`）和距离项（`dist`），实现：
  - 近距离：高斯项接近1，magnitude主要由距离项控制，提供线性增长
  - 远距离：高斯项快速衰减，magnitude整体衰减，避免远距离原子的干扰
- 这种设计平衡了局部性和全局性，既保证了收敛稳定性，又避免了过度局部化

**8. Tanh (最终选择)** - 决赛圈方法：

```
dist_ij = ||query_i - atom_j||
w_softmax = softmax(-dist_ij / sig_sf)  // 方向选择：选择最近的原子
w_mag = tanh(dist_ij / sig_mag)  // Tanh函数控制magnitude
diff_normed = (atom_j - query_i) / ||atom_j - query_i||  // 归一化方向
vector_field[i, type_j] = Σ(w_softmax · w_mag · diff_normed)
```

**设计思路**：
- `w_softmax`: 与gaussian_mag相同，使用softmax进行方向选择
- `w_mag`: 使用tanh函数，提供平滑的sigmoid形状的magnitude控制：
  - **近距离**（`dist << sig_mag`）: `tanh(x) ≈ x`，magnitude近似线性增长
  - **中等距离**（`dist ≈ sig_mag`）: `tanh(x) ≈ 0.76`，magnitude达到饱和
  - **远距离**（`dist >> sig_mag`）: `tanh(x) → 1`，magnitude饱和，但保持有界
- Tanh函数的优势：
  - **有界性**：输出范围`[0, 1]`，避免数值爆炸
  - **平滑性**：连续可微，梯度稳定
  - **自适应性**：通过`sig_mag`参数灵活控制饱和点

**方法选择与实验验证**：

通过系统性的实验对比，我们在QM9数据集上评估了所有场定义方法的重建效果（RMSD、原子数量匹配率、收敛稳定性等）。最终，**gaussian_mag**和**tanh**两种复合方法进入了"决赛圈"。

**对比结果**（基于QM9数据集）：
- **Tanh方法**在RMSD指标上表现更优（平均RMSD: 0.167 vs 0.721），重建精度更高
- **Tanh方法**在重建时间上更高效（平均时间: 41.4s vs 57.9s）
- **Tanh方法**的数值稳定性更好，避免了gaussian_mag在某些情况下可能出现的数值问题

因此，**最终选择tanh方法**作为默认的场定义方法。该方法在配置文件中的典型参数为：
- `sig_sf = 0.1`: 控制softmax的局部性
- `sig_mag = 2.0`: 控制tanh的饱和点（对于QM9数据集）

**设计哲学**：

复合场定义方法的核心创新在于**分离关注点**：
1. **方向选择**（`w_softmax`）：使用softmax确保选择最相关的原子，提供稳定的方向指引
2. **模长控制**（`w_mag`）：独立控制梯度强度，影响收敛速度和稳定性
3. **归一化方向**（`diff_normed`）：保证方向向量的单位长度，避免距离对方向的影响

这种设计使得场定义既具有理论上的优雅性（分离方向与模长），又具有实践上的有效性（在重建任务中表现优异）。

#### 2.5.2 场到分子（gnf2mol）

**输入**：
- 解码器模型
- 潜在码：`[B, grid_size³, code_dim]`

**输出**：
- 原子坐标：`[B, N_atoms, 3]`
- 原子类型：`[B, N_atoms]`

**算法**（梯度上升）：

1. **初始化**：为每个原子类型随机初始化`n_query_points`个采样点

2. **迭代优化**（`n_iter`步）：
   ```
   for iter in range(n_iter):
       # 计算当前点的矢量场
       field = decoder(query_points, codes)  # [B, n_points, n_atom_types, 3]
       
       # 选择对应原子类型的梯度
       grad = field[point_idx, atom_type_idx, :]  # [n_points, 3]
       
       # 梯度上升
       query_points ← query_points + step_size · grad
       
       # 检查收敛（可选）
       if ||grad|| < threshold:
           break
   ```

3. **聚类**：使用DBSCAN对最终点进行聚类，得到原子位置

4. **原子类型分配**：根据点的初始原子类型分配

## 3. DDPM扩散过程

### 3.1 前向扩散

**定义**：
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t) I)
```

其中：
- `α_t = 1 - β_t`
- `ᾱ_t = Π_{s=1}^t α_s`
- `β_t`是噪声调度（linear或cosine）

**采样**：
```
x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε,  ε ~ N(0, I)
```

### 3.2 反向采样
预测原始数据（x0）

**单步采样**：
```
predicted_x0 = model(x_t, t)  # 直接预测x0
ε_θ = (x_t - √(ᾱ_t) · predicted_x0) / √(1 - ᾱ_t)  # 推导噪声
x_{t-1} = (1/√(α_t)) · (x_t - (β_t/√(1 - ᾱ_t)) · ε_θ) + √(σ_t²) · z
```

### 3.3 训练损失
**预测x0版本**：
```
L = E_{t,x_0,ε} [||x_0 - x_0_θ(x_t, t)||²]
```

## 4. 数据流

### 4.1 训练阶段

**神经场训练**：
```
分子 → 编码器 → 潜在码 → 解码器 → 矢量场 → 损失（与GT场比较）
```

**FuncMol训练**：
```
分子 → 编码器 → 潜在码 → 加噪声 → 去噪器 → 预测 → 损失（与原始码比较）
```

### 4.2 生成阶段

```
随机噪声 → DDPM采样 → 潜在码 → 解码器 → 矢量场 → 梯度上升 → 分子结构
```

## 5. 关键设计选择

1. **潜在码空间**：在低维潜在空间（`grid_size³ × code_dim`）生成，而非直接生成原子坐标
2. **连续场表示**：使用矢量场而非离散图，支持任意分辨率查询
3. **几何等变性**：EGNN保证旋转和平移等变性
4. **多原子类型**：为每种原子类型独立建模矢量场
5. **局部性**：通过k-NN和cutoff保证空间局部性，提高效率

