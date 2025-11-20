# Figure 2 详细架构设计

## 整体布局

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Figure 2: Detailed Architecture                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────────────────────────┐  ┌──────────────────────────────┐ │
│  │  Neural Field Autoencoder      │  │  Diffusion Model             │ │
│  │  (Left Panel)                  │  │  (Right Panel)               │ │
│  └────────────────────────────────┘  └──────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 左半部分：Neural Field Autoencoder

### 布局结构（从上到下）

```
[Molecule] 
    ↓
[Field Representation]
    ↓
[CrossGraph Encoder]
    ├─ [Atom Graph] ──k-NN──> [Grid Anchors]
    └─ [Message Passing] ──> [Latent Codes]
    ↓
[3D Latent Codes Grid]
```

### 详细设计

#### 1. 输入：Molecule（顶部）
- **可视化**：3D球棍模型
- **标注**：原子类型（C, H, O, N, F）用不同颜色
- **示例**：使用一个小分子（如甲烷或甲醇）

#### 2. Field Representation（Molecule → Field）
- **可视化**：
  - 左侧：分子结构（球棍模型）
  - 右侧：矢量场可视化（箭头图）
  - 中间：箭头表示转换过程
- **标注**："Molecule to Field" 或 "Field Conversion"
- **细节**：可以展示一个2D切片或3D视角的场

#### 3. CrossGraph Encoder（核心部分）

**子图A：图结构构建**
```
┌─────────────────────────────────────┐
│  CrossGraph Structure               │
├─────────────────────────────────────┤
│                                     │
│  [Atoms] ──k-NN (atom_k=8)──> [Atoms]  │
│    │                                  │
│    │ k-NN (k=32)                      │
│    ↓                                  │
│  [Grid Anchors] ──k-NN──> [Grid Anchors] │
│                                     │
│  Legend:                            │
│  • Atoms (colored by type)          │
│  • Grid Anchors (gray cubes)        │
│  • k-NN connections (dashed lines) │
└─────────────────────────────────────┘
```

**可视化元素**：
- **原子节点**：用彩色球表示（C=灰，H=白，O=红，N=蓝，F=绿）
- **网格锚点**：用灰色小立方体表示，排列成8×8×8网格
- **连接线**：
  - 原子-原子连接：实线（atom_k_neighbors=8）
  - 原子-网格连接：虚线（k_neighbors=32），从每个网格锚点指向最近的原子
- **标注**：
  - "Atom Graph"：标注原子内部图
  - "Cross-Graph Connections"：标注原子-网格连接
  - "k-NN (k=32)"：标注连接参数

**子图B：消息传递过程**
```
┌─────────────────────────────────────┐
│  Message Passing (GNN Layers)      │
├─────────────────────────────────────┤
│                                     │
│  Layer 1:                           │
│  [h_atom, h_grid] ──> [m_ij] ──> [h'_atom, h'_grid] │
│                                     │
│  Layer 2:                           │
│  [h'_atom, h'_grid] ──> [m'_ij] ──> [h''_atom, h''_grid] │
│                                     │
│  ...                                │
│                                     │
│  Layer L:                           │
│  Extract grid features ──> [Codes]  │
│                                     │
│  Message Function:                  │
│  m_ij = MLP([h_i, h_j, dist_emb])  │
│                                     │
│  Aggregation:                       │
│  h'_i = h_i + LayerNorm(Σ m_ij)    │
└─────────────────────────────────────┘
```

**可视化元素**：
- 使用流程图展示多层消息传递
- 标注消息函数和聚合函数
- 显示残差连接

#### 4. 输出：3D Latent Codes Grid（底部）
- **可视化**：
  - 3D网格结构（8×8×8 = 512个锚点）
  - 每个网格单元用颜色表示code值（使用热力图）
  - 可以展示一个切面视图
- **标注**：
  - "Latent Codes [8³, code_dim]"
  - "Grid Size: 8×8×8"
  - "Code Dimension: 128"

### 左半部分完整流程

```
┌─────────────────────────────────────────────────────────────┐
│  Neural Field Autoencoder                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Molecule]                                                 │
│    ↓                                                        │
│  [Field] (Vector Field Visualization)                      │
│    ↓                                                        │
│  ┌──────────────────────────────────────┐                 │
│  │  CrossGraph Encoder                   │                 │
│  │  ┌──────────────────────────────────┐ │                 │
│  │  │  Graph Construction              │ │                 │
│  │  │  • Atoms (colored)                │ │                 │
│  │  │  • Grid Anchors (8×8×8)          │ │                 │
│  │  │  • k-NN Connections               │ │                 │
│  │  └──────────────────────────────────┘ │                 │
│  │  ┌──────────────────────────────────┐ │                 │
│  │  │  Message Passing (L layers)      │ │                 │
│  │  │  • Edge Features: [h_i, h_j, d]  │ │                 │
│  │  │  • Aggregation: Mean              │ │                 │
│  │  │  • Update: h + LayerNorm(agg)    │ │                 │
│  │  └──────────────────────────────────┘ │                 │
│  └──────────────────────────────────────┘                 │
│    ↓                                                        │
│  [Latent Codes] [8³, 128]                                  │
│  (3D Grid Visualization)                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 右半部分：Diffusion Model

### 布局结构（从上到下）

```
[Codes] (Input)
    ↓
[Diffusion Process]
    ├─ [Forward Diffusion] (Training)
    └─ [Reverse Sampling] (Inference)
    ↓
[New Codes] (Output)
```

### 详细设计

#### 1. 输入：Codes（顶部）
- **可视化**：与左半部分相同的3D网格结构
- **标注**："Input Codes [8³, 128]"

#### 2. Diffusion Process（核心部分）

**子图A：Forward Diffusion（训练阶段）**
```
┌─────────────────────────────────────┐
│  Forward Diffusion (Training)       │
├─────────────────────────────────────┤
│                                     │
│  z₀ ──Add Noise──> z_t ──> z_T      │
│  (clean)    q(x_t|x_0)    (noise)  │
│                                     │
│  z₀ = [8³, 128]                     │
│  z_t = √(ᾱ_t) z₀ + √(1-ᾱ_t) ε     │
│  z_T ~ N(0, I)                      │
│                                     │
│  Time Steps: t = 0, 1, ..., T      │
└─────────────────────────────────────┘
```

**可视化元素**：
- 展示从干净码到噪声的渐变过程
- 使用颜色映射显示噪声水平
- 可以展示几个关键时间步（t=0, t=T/2, t=T）

**子图B：Denoiser Architecture**
```
┌─────────────────────────────────────┐
│  Denoiser (EGNN + Time Embedding)  │
├─────────────────────────────────────┤
│                                     │
│  [Noisy Codes]                      │
│    ↓                                │
│  [Input Projection]                 │
│    ↓                                │
│  [Time Embedding] ──> [t_emb]      │
│    ↓                                │
│  ┌──────────────────────────────┐  │
│  │  EGNN Layers (L layers)       │  │
│  │  • Graph: radius_graph        │  │
│  │  • Message: [h_i, h_j, d, t] │  │
│  │  • Update: h + MLP([h, m])   │  │
│  └──────────────────────────────┘  │
│    ↓                                │
│  [Output Projection]                │
│    ↓                                │
│  [Predicted] (ẑ₀ or ε̂)             │
│                                     │
└─────────────────────────────────────┘
```

**可视化元素**：
- 展示EGNN层的结构
- 标注时间嵌入的作用
- 显示图构建过程（radius_graph）

**子图C：Reverse Sampling（推理阶段）**
```
┌─────────────────────────────────────┐
│  Reverse Sampling (Inference)       │
├─────────────────────────────────────┤
│                                     │
│  z_T ──Denoise──> z_{T-1} ──> ... ──> z₁ ──> z₀ │
│  (noise)  p(x_{t-1}|x_t)           (clean)      │
│                                     │
│  z_{t-1} = μ_θ(z_t, t) + σ_t · ε   │
│                                     │
│  Time Steps: t = T, T-1, ..., 1, 0 │
└─────────────────────────────────────┘
```

**可视化元素**：
- 展示从噪声到干净码的去噪过程
- 使用颜色映射显示去噪进度
- 展示几个关键时间步

#### 3. 输出：New Codes（底部）
- **可视化**：与输入相同的3D网格结构
- **标注**："Generated Codes [8³, 128]"

### 右半部分完整流程

```
┌─────────────────────────────────────────────────────────────┐
│  Diffusion Model                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Input Codes] [8³, 128]                                    │
│    ↓                                                        │
│  ┌──────────────────────────────────────┐                 │
│  │  Forward Diffusion (Training)         │                 │
│  │  z₀ ──> z_t ──> z_T                  │                 │
│  │  q(x_t|x_0) = N(√(ᾱ_t)z₀, (1-ᾱ_t)I) │                 │
│  └──────────────────────────────────────┘                 │
│    ↓                                                        │
│  ┌──────────────────────────────────────┐                 │
│  │  Denoiser (EGNN + Time)              │                 │
│  │  • Time Embedding: Sinusoidal        │                 │
│  │  • Graph: radius_graph(r)            │                 │
│  │  • Message: [h_i, h_j, d, t_emb]     │                 │
│  │  • Output: ẑ₀ or ε̂                  │                 │
│  └──────────────────────────────────────┘                 │
│    ↓                                                        │
│  ┌──────────────────────────────────────┐                 │
│  │  Reverse Sampling (Inference)        │                 │
│  │  z_T ──> z_{T-1} ──> ... ──> z₀     │                 │
│  │  p(x_{t-1}|x_t) = N(μ_θ, σ_t²)      │                 │
│  └──────────────────────────────────────┘                 │
│    ↓                                                        │
│  [Generated Codes] [8³, 128]                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 关键可视化元素

### 1. 3D网格可视化（Codes）

**方法1：体素渲染**
- 使用3D体素图，每个体素用颜色表示code值
- 可以展示切面视图（XY, XZ, YZ平面）

**方法2：点云可视化**
- 每个网格锚点用一个点表示
- 颜色表示code的某个维度或模长

**方法3：热力图切片**
- 展示2D切片（例如中间层）
- 使用热力图颜色映射

### 2. 图结构可视化

**原子-网格连接**：
- 使用3D散点图显示原子和网格锚点
- 用线条连接（k-NN连接）
- 可以只显示部分连接以避免过于密集

**建议**：
- 使用透明度控制连接线的可见性
- 只显示每个网格锚点的前k个连接
- 使用不同颜色区分不同类型的连接

### 3. 扩散过程可视化

**时间步序列**：
- 展示5-7个关键时间步
- 使用颜色映射显示噪声水平
- 从蓝色（干净）到红色（噪声）

**去噪过程**：
- 展示从噪声到干净的渐变
- 可以展示预测的中间状态

## 标注和文字说明

### 左半部分标注

1. **"Molecule"**：标注输入分子
2. **"Field Conversion"**：标注转换过程
3. **"CrossGraph Encoder"**：标注编码器
4. **"k-NN Connections"**：标注连接方式
5. **"Message Passing"**：标注消息传递
6. **"Latent Codes"**：标注输出

### 右半部分标注

1. **"Input Codes"**：标注输入
2. **"Forward Diffusion"**：标注前向过程
3. **"Denoiser"**：标注去噪器
4. **"Time Embedding"**：标注时间嵌入
5. **"Reverse Sampling"**：标注反向采样
6. **"Generated Codes"**：标注输出

### 关键参数标注

- **Grid Size**: 8×8×8 = 512 anchors
- **Code Dimension**: 128
- **k-NN (atom)**: 8
- **k-NN (grid)**: 32
- **GNN Layers**: 4-6
- **Diffusion Steps**: T = 100 or 1000
- **Time Embedding Dim**: 64

## 颜色方案

### 左半部分
- **分子原子**：CPK颜色（C=灰，H=白，O=红，N=蓝，F=绿）
- **网格锚点**：灰色或浅蓝色
- **连接线**：浅灰色（虚线）
- **场箭头**：viridis颜色映射

### 右半部分
- **Codes网格**：使用热力图（蓝色=低值，红色=高值）
- **扩散过程**：渐变（蓝色=干净，红色=噪声）
- **去噪过程**：渐变（红色=噪声，蓝色=干净）

## 技术实现建议

### 绘图工具
- **推荐**：TikZ (LaTeX), Inkscape, Adobe Illustrator
- **Python**：matplotlib + mpl_toolkits.mplot3d（用于3D可视化）

### 子图组织
建议使用2×2或3×2的子图布局：

```
┌─────────────┬─────────────┐
│  Left Top   │ Right Top   │
│  (Encoder)  │ (Diffusion) │
├─────────────┼─────────────┤
│  Left Bot   │ Right Bot   │
│  (Details)  │ (Details)   │
└─────────────┴─────────────┘
```

### 分辨率要求
- **论文**：至少 300 DPI
- **格式**：PDF（矢量图）或 PNG（高分辨率位图）

## 简化版本（如果空间有限）

如果空间有限，可以只展示核心元素：

**左半部分**：
- Molecule → Field → Encoder → Codes
- 重点展示CrossGraph连接

**右半部分**：
- Codes → Denoiser → New Codes
- 重点展示扩散过程的关键步骤





