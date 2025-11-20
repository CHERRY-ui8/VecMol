# 流程图变量维度说明

## 完整流程维度详解

### 1. 分子数据 (3D坐标 + 原子类型)

**原始格式**（单个分子）：
- 原子坐标：`[N_atoms, 3]`
  - `N_atoms`: 分子中的原子数量（可变，不同分子不同）
  - `3`: 3D坐标 (x, y, z)
- 原子类型：`[N_atoms]`
  - 每个原子对应一个类型索引（0到`n_atom_types-1`）

**编码器输入格式**：

**方式1：PyTorch Geometric Batch格式**（当前实现）：
- 当多个分子组成batch时，使用PyTorch Geometric的Batch格式：
  - `data.pos`: `[N_total_atoms, 3]` - 所有分子的原子坐标拼接在一起
  - `data.x`: `[N_total_atoms]` - 所有分子的原子类型拼接在一起
  - `data.batch`: `[N_total_atoms]` - 每个原子属于哪个分子的索引
  - 详见第3节编码器输入格式的详细说明

**方式2：固定大小的Dense Tensor格式**（如果不使用PyG）：
- 如果使用固定大小的张量，需要padding到最大原子数：
  - 原子坐标：`[B, max_atoms, 3]`
    - `B`: batch size
    - `max_atoms`: batch中所有分子的最大原子数（需要padding）
    - `3`: 3D坐标
  - 原子类型：`[B, max_atoms]`
    - 每个原子对应一个类型索引（padding位置使用特殊值，如`PADDING_INDEX=-1`）
  - 原子mask（可选）：`[B, max_atoms]`
    - 布尔值，标识哪些位置是真实原子，哪些是padding
- **注意**：`[Batch, atom_type, n_atoms, 3]` 格式不正确，应该是分开的两个张量

**典型值**：
- QM9数据集：`N_atoms` 通常在 5-29 之间
- `n_atom_types`: 5（C, N, O, F, H）

---

### 2. [数据预处理] → 转换为场表示

**场表示格式**：
- 查询点：`[B, n_points, 3]`
  - `B`: batch size（批次大小）
  - `n_points`: 查询点数量（通常为500-4000，取决于数据集）
  - `3`: 3D坐标
- 矢量场：`[B, n_points, n_atom_types, 3]`
  - `n_atom_types`: 原子类型数量（通常为5）
  - `3`: 每个查询点对每个原子类型预测一个3D向量

**典型值**：
- QM9: `n_points = 500`
- CREMP: `n_points = 4000`

---

### 3. [神经场编码器] → 潜在码 codes

**输入格式**：

**方式1：PyTorch Geometric Batch对象**（当前实现）：
- `data.pos`: `[N_total_atoms, 3]` - 所有batch中所有原子的坐标（拼接在一起）
  - `N_total_atoms`: 一个batch中所有分子的原子总数（`N_total_atoms = Σ N_atoms_i`，i为batch中的分子索引）
  - `3`: 3D坐标 (x, y, z)
- `data.x`: `[N_total_atoms]` - 所有batch中所有原子的类型（拼接在一起）
  - 每个原子对应一个类型索引（0到`n_atom_types-1`）
  - **关键**：通过**索引对应**关系，`data.pos[i]` 和 `data.x[i]` 表示同一个原子的坐标和类型
- `data.batch`: `[N_total_atoms]` - 每个原子属于哪个分子的batch索引
  - 例如：如果有3个分子，分别有5、8、6个原子，则`batch = [0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2]`
  - 用于区分哪些原子属于哪个分子

**如何同时包含坐标和类型信息？**
- **通过索引对应**：`data.pos` 和 `data.x` 的第一个维度都是 `N_total_atoms`，通过相同的索引 `i` 来匹配
  - `data.pos[i]` → 第 `i` 个原子的坐标 `[3]`
  - `data.x[i]` → 第 `i` 个原子的类型索引（标量）
- **示例**：
  ```python
  # 假设有2个原子
  data.pos = [[0.0, 0.0, 0.0],  # 第0个原子的坐标
              [1.0, 0.0, 0.0]]  # 第1个原子的坐标
  data.x = [0, 1]  # 第0个原子是类型0（如C），第1个原子是类型1（如N）
  # 通过索引对应：pos[0] 和 x[0] 属于同一个原子
  ```

**输入示例**：
```python
# 单个分子（batch_size=1）
data = Data(
    pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], ...]),  # [N_atoms, 3]
    x=torch.tensor([0, 1, 2, ...]),  # [N_atoms] - 原子类型索引
    batch=torch.zeros(N_atoms, dtype=torch.long)  # [N_atoms] - 全为0表示都是第0个分子
)

# 多个分子组成batch（batch_size=B）
# PyTorch Geometric的DataLoader会自动将多个Data对象合并成Batch
batch = Batch.from_data_list([data1, data2, ..., dataB])
# batch.pos: [N_total_atoms, 3] - 所有原子的坐标拼接
# batch.x: [N_total_atoms] - 所有原子的类型拼接
# batch.batch: [N_total_atoms] - 每个原子属于哪个分子的索引
```

**方式2：固定大小的Dense Tensor格式**（如果不使用PyG）：
```python
# 使用 to_dense_batch 转换（PyG提供的工具函数）
from torch_geometric.utils import to_dense_batch

# 从PyG Batch转换
coords, atom_mask = to_dense_batch(batch.pos, batch.batch, fill_value=0)
atoms_channel, _ = to_dense_batch(batch.x, batch.batch, fill_value=PADDING_INDEX)
# coords: [B, max_atoms, 3]
# atoms_channel: [B, max_atoms]
# atom_mask: [B, max_atoms] - 布尔mask，标识真实原子

# 或者直接提供固定大小的输入
coords = torch.tensor([...])  # [B, max_atoms, 3]
atoms_channel = torch.tensor([...])  # [B, max_atoms]
atom_mask = torch.tensor([...])  # [B, max_atoms] (可选)
```

**注意**：
- 固定大小格式需要padding到`max_atoms`（batch中最大原子数）
- 需要mask来区分真实原子和padding
- 当前encoder实现只支持PyG Batch格式，如需支持dense格式需要修改encoder

**内部处理**（如何分辨不同原子种类）：

1. **提取坐标和类型**：
   ```python
   atom_coords = data.pos      # [N_total_atoms, 3] - 坐标
   atoms_channel = data.x      # [N_total_atoms] - 类型索引
   ```
   - 通过相同的索引 `i`，`atom_coords[i]` 和 `atoms_channel[i]` 属于同一个原子

2. **原子类型转换为one-hot编码**：
   ```python
   atom_feat = F.one_hot(atoms_channel.long(), num_classes=self.n_atom_types).float()
   # [N_total_atoms, n_atom_types]
   ```
   - 将类型索引转换为one-hot向量
   - 例如：类型0（C）→ `[1, 0, 0, 0, 0]`，类型1（N）→ `[0, 1, 0, 0, 0]`

3. **填充到code_dim维度**：
   ```python
   if self.n_atom_types < self.code_dim:
       padding = torch.zeros(N_total_atoms, self.code_dim - self.n_atom_types, device=device)
       atom_feat = torch.cat([atom_feat, padding], dim=1)  # [N_total_atoms, code_dim]
   ```
   - 如果`code_dim > n_atom_types`，用0填充剩余维度

4. **构建节点特征和坐标**：
   ```python
   node_feats = torch.cat([atom_feat, grid_codes], dim=0)  # 原子特征 + 网格特征
   node_pos = torch.cat([atom_coords, grid_coords_flat], dim=0)  # 原子坐标 + 网格坐标
   ```
   - 每个节点（原子或网格锚点）都有：
     - **特征向量**：`node_feats[i]` - 包含原子类型信息（one-hot编码）
     - **坐标向量**：`node_pos[i]` - 包含空间位置信息

5. **构建图结构**（原子-原子连接 + 网格-原子连接）

6. **通过GNN消息传递更新特征**：
   - GNN在消息传递时会同时使用：
     - 节点特征（包含原子类型信息）
     - 节点坐标（包含空间位置信息）
     - 边特征（距离编码）

**总结**：
- **坐标信息**：存储在 `data.pos` 中，通过索引访问
- **类型信息**：存储在 `data.x` 中，通过**相同的索引**与坐标对应
- **编码过程**：类型索引 → one-hot编码 → 特征向量，与坐标一起输入GNN
- **分辨方式**：通过one-hot编码的维度位置来区分不同原子类型（如第0维=1表示C，第1维=1表示N）

**输出格式**：
- 潜在码：`[B, grid_size³, code_dim]`
  - `B`: batch size（从`data.num_graphs`获取）
  - `grid_size³`: 锚点网格的总点数（`grid_size × grid_size × grid_size`）
  - `code_dim`: 潜在码的特征维度

**重要说明**：
- **latent codes 没有 `n_atom_types` 维度**
- codes 是原子类型无关的（type-agnostic）表示
- `n_atom_types` 维度是在解码器阶段才引入的
- 解码器会为每个查询点预测所有原子类型的矢量场（见第6节）

**典型值**：
- `grid_size`: 
  - QM9: 9 → `grid_size³ = 729`
  - DRUGS: 11 → `grid_size³ = 1331`
  - CREMP: 24 → `grid_size³ = 13824`
- `code_dim`: 128 或 1024（通常为128）

**注意**：用户流程图中写的是 `[B, N³, code_dim]`，其中 `N = grid_size`

---

### 4. [生成模型训练] → 学习 codes 的分布

**输入/输出格式**：
- 输入（带噪声的codes）：`[B, grid_size³, code_dim]`
- 时间步：`[B]`（用于DDPM）
- 输出（预测的噪声或原始数据）：`[B, grid_size³, code_dim]`

**训练过程**：
- 前向扩散：`x_0` → `x_t`，维度保持 `[B, grid_size³, code_dim]`
- 去噪预测：`x_t, t` → `ε_θ` 或 `x_0_θ`，维度保持 `[B, grid_size³, code_dim]`

---

### 5. [采样生成] → 新的 codes

**输出格式**：
- 生成的潜在码：`[B, grid_size³, code_dim]`
  - 维度与编码器输出相同

**采样过程**：
- 初始噪声：`[B, grid_size³, code_dim]`（从标准正态分布采样）
- 经过T步DDPM去噪后：`[B, grid_size³, code_dim]`

---

### 6. [神经场解码器] → 重建分子场

**输入格式**：
- 查询点：`[B, n_points, 3]`
- 潜在码：`[B, grid_size³, code_dim]`
  - **注意**：codes 没有 `n_atom_types` 维度，是原子类型无关的表示

**输出格式**：
- 矢量场：`[B, n_points, n_atom_types, 3]`
  - 每个查询点对每个原子类型预测一个3D向量
  - `n_atom_types` 维度是在解码器的最后一层（`field_layer`）引入的
  - 通过设置 `out_x_dim=n_atom_types`，为每个节点预测所有原子类型的坐标

**内部处理**：
1. 将 codes 作为网格锚点的初始特征：`[B, grid_size³, code_dim]`
2. 查询点初始化为零特征：`[B, n_points, code_dim]`
3. 通过 k-NN 连接查询点到最近的网格锚点
4. 多层 EGNN 消息传递更新特征
5. 最后一层 EGNN（`field_layer`）预测 `n_atom_types` 个坐标：`[B, n_points, n_atom_types, 3]`
6. 计算相对于查询点的残差向量，得到矢量场

---

### 7. [场到分子转换] → 生成3D分子结构

**输入格式**：
- 矢量场：`[B, n_points, n_atom_types, 3]`
- 初始采样点：`[n_query_points, 3]`（为每个原子类型初始化）

**输出格式**：
- 原子坐标：`[N_atoms, 3]` 或 batch中 `[B, N_atoms, 3]`
  - `N_atoms`: 生成的原子数量（可变）
- 原子类型：`[N_atoms]` 或 batch中 `[B, N_atoms]`

**转换过程**（梯度上升）：
1. 初始化：为每个原子类型随机采样 `n_query_points` 个点
2. 迭代优化：使用矢量场进行梯度上升，更新采样点位置
3. 聚类：使用DBSCAN对最终点进行聚类，得到原子位置
4. 类型分配：根据点的初始原子类型分配

---

## 关键维度参数总结

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `B` | Batch size（批次大小） | 训练时：32-128，采样时：10-100 |
| `N_atoms` | 分子中的原子数量 | QM9: 5-29，可变 |
| `n_atom_types` | 原子类型数量 | QM9: 5（C, N, O, F, H） |
| `n_points` | 查询点数量 | QM9: 500，CREMP: 4000 |
| `grid_size` | 锚点网格大小 | QM9: 9，DRUGS: 11，CREMP: 24 |
| `grid_size³` | 锚点总数 | QM9: 729，DRUGS: 1331，CREMP: 13824 |
| `code_dim` | 潜在码维度 | 128 或 1024（通常为128） |

---

## 维度变化流程图

```
分子数据
├─ 原子坐标: [B, N_atoms, 3]
└─ 原子类型: [B, N_atoms]
    ↓
[数据预处理]
    ↓
场表示
├─ 查询点: [B, n_points, 3]
└─ 矢量场: [B, n_points, n_atom_types, 3]
    ↓
[神经场编码器]
    ↓
潜在码: [B, grid_size³, code_dim]
    ↓
[生成模型训练/采样]
    ↓
新的潜在码: [B, grid_size³, code_dim]
    ↓
[神经场解码器]
    ↓
重建矢量场: [B, n_points, n_atom_types, 3]
    ↓
[场到分子转换]
    ↓
生成的分子
├─ 原子坐标: [B, N_atoms, 3]
└─ 原子类型: [B, N_atoms]
```

---

## 注意事项

1. **可变维度**：
   - `N_atoms` 在不同分子间是变化的，因此通常使用padding或图结构处理
   - 在batch处理时，需要处理不同大小的分子

2. **固定维度**：
   - `grid_size³` 和 `code_dim` 是固定的，由模型配置决定
   - 这些维度在整个流程中保持一致

3. **维度匹配**：
   - 编码器和解码器必须使用相同的 `grid_size` 和 `code_dim`
   - 生成模型（DDPM）的输入输出维度必须与编码器输出维度匹配

