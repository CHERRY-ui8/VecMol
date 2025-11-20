# FuncMol 算法伪代码

本文档提供FuncMol模型的论文级别算法伪代码，适用于学术论文撰写。

---

## Algorithm 1: 神经场自编码器训练

**输入**：分子数据集 $\mathcal{D} = \{(X_i, T_i)\}_{i=1}^N$，其中 $X_i \in \mathbb{R}^{n_i \times 3}$ 是原子坐标，$T_i \in \{0,1,\ldots,K-1\}^{n_i}$ 是原子类型

**输出**：编码器 $E_\phi$ 和解码器 $D_\psi$

```
1: 初始化编码器 $E_\phi$ 和解码器 $D_\psi$
2: 创建锚点网格 $\mathcal{G} = \{g_j\}_{j=1}^{L^3}$，其中 $L$ 是网格大小
3: 
4: for epoch = 1 to max_epochs do
5:     for batch $(X, T) \sim \mathcal{D}$ do
6:         // 编码阶段
7:         $z = E_\phi(X, T)$  // $z \in \mathbb{R}^{B \times L^3 \times d}$
8:         
9:         // 采样查询点
10:        $Q \sim \mathcal{U}(\mathcal{B})$  // 在边界框内均匀采样 $n_q$ 个点
11:        
12:        // 解码阶段
13:        $\hat{V} = D_\psi(Q, z)$  // $\hat{V} \in \mathbb{R}^{B \times n_q \times K \times 3}$
14:        
15:        // 计算真实场
16:        $V^* = \text{GNFConverter}(X, T, Q)$  // 使用公式计算GT场
17:        
18:        // 计算损失
19:        $\mathcal{L} = \frac{1}{B \cdot n_q \cdot K} \sum_{b,i,k} \|\hat{V}_{b,i,k} - V^*_{b,i,k}\|^2$
20:        
21:        // 反向传播
22:        $\nabla_\phi \mathcal{L}$, $\nabla_\psi \mathcal{L}$ ← backprop($\mathcal{L}$)
23:        update($\phi$, $\nabla_\phi \mathcal{L}$)
24:        update($\psi$, $\nabla_\psi \mathcal{L}$)
25:    end for
26: end for
27: 
28: return $E_\phi$, $D_\psi$
```

**编码器 $E_\phi$ 的详细过程**：

```
function E_\phi(X, T):
    // 初始化节点特征
    $h_a^{(0)} = \text{OneHot}(T)$  // 原子节点特征
    $h_g^{(0)} = \mathbf{0}$        // 网格节点特征（初始化为零）
    
    // 构建图
    $\mathcal{E}_{aa} = \text{kNN}(X, k_{atom})$  // 原子内部连接
    $\mathcal{E}_{ag} = \text{kNN}(X \to \mathcal{G}, k_{grid})$  // 原子-网格连接
    
    // 消息传递
    for $l = 1$ to $L_{enc}$ do
        // 消息计算
        for $(i,j) \in \mathcal{E}_{aa} \cup \mathcal{E}_{ag}$ do
            $d_{ij} = \|x_i - x_j\|$
            $e_{ij} = \text{GaussianSmearing}(d_{ij})$
            $m_{ij} = \text{MLP}_e([h_i, h_j, e_{ij}])$
        end for
        
        // 聚合和更新
        for each node $i$ do
            $m_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}$
            $h_i^{(l)} = h_i^{(l-1)} + \text{LayerNorm}(\text{MLP}_h([h_i^{(l-1)}, m_i]))$
        end for
    end for
    
    // 提取网格特征作为潜在码
    $z = \{h_g^{(L_{enc})}\}_{g \in \mathcal{G}}$
    return $z$
end function
```

**解码器 $D_\psi$ 的详细过程**（EGNN）：

```
function D_\psi(Q, z):
    // 初始化
    $h_q^{(0)} = \mathbf{0}$  // 查询点特征
    $h_g^{(0)} = z$           // 网格锚点特征
    $x_q = Q$                 // 查询点坐标
    $x_g = \mathcal{G}$       // 网格锚点坐标
    
    // 构建图
    $\mathcal{E}_{qg} = \text{kNN}(Q \to \mathcal{G}, k_{neighbors})$
    
    // EGNN消息传递
    for $l = 1$ to $L_{dec}$ do
        for $(i,j) \in \mathcal{E}_{qg}$ do
            // 计算相对位置和距离
            $\mathbf{r}_{ij} = x_j - x_i$
            $d_{ij} = \|\mathbf{r}_{ij}\|$
            
            // 消息计算
            $m_{ij} = \text{MLP}_e([h_i, h_j, d_{ij}])$
            
            // Cutoff函数
            $C_{ij} = \frac{1}{2}(\cos(\pi d_{ij} / r_{cut}) + 1) \cdot \mathbf{1}[d_{ij} \leq r_{cut}]$
            $m_{ij} = m_{ij} \odot C_{ij}$
        end for
        
        // 坐标更新（等变）
        for each query point $i$ do
            $a_{ij} = \text{MLP}_x(m_{ij})$  // 标量系数
            $\Delta x_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} a_{ij} \cdot \frac{\mathbf{r}_{ij}}{d_{ij} + \epsilon}$
            $x_i^{(l)} = x_i^{(l-1)} + \Delta x_i$
        end for
        
        // 特征更新
        for each node $i$ do
            $m_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}$
            $h_i^{(l)} = h_i^{(l-1)} + \text{MLP}_h([h_i^{(l-1)}, m_i])$
        end for
    end for
    
    // 场预测
    $s_i^{(k)} = \text{MLP}_{field}(h_i^{(L_{dec})})$  // 预测 $K$ 个源点坐标
    $\mathbf{v}_i^{(k)} = s_i^{(k)} - x_i$  // 计算相对于查询点的向量
    return $\mathbf{V} = \{\mathbf{v}_i^{(k)}\}_{i,k}$
end function
```

---

## Algorithm 2: DDPM训练（FuncMol去噪器）

**输入**：预训练的编码器 $E_\phi$，潜在码数据集 $\mathcal{Z} = \{z_i\}_{i=1}^N$，其中 $z_i = E_\phi(X_i, T_i)$

**输出**：去噪器 $f_\theta$

```
1: 初始化去噪器 $f_\theta$
2: 定义噪声调度 $\{\beta_t\}_{t=1}^T$，计算 $\{\alpha_t\}$, $\{\bar{\alpha}_t\}$
3: 
4: for epoch = 1 to max_epochs do
5:     for batch $z \sim \mathcal{Z}$ do
6:         // 随机采样时间步
7:         $t \sim \mathcal{U}(\{1, \ldots, T\})$
8:         
9:         // 采样噪声
10:        $\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
11:        
12:        // 前向扩散
13:        $z_t = \sqrt{\bar{\alpha}_t} z + \sqrt{1 - \bar{\alpha}_t} \varepsilon$
14:        
15:        // 预测（两种变体）
16:        if method == "predict_epsilon" then
17:            $\hat{\varepsilon} = f_\theta(z_t, t)$
18:            $\mathcal{L} = \|\varepsilon - \hat{\varepsilon}\|^2$
19:        else if method == "predict_x0" then
20:            $\hat{z}_0 = f_\theta(z_t, t)$
21:            $\mathcal{L} = \|z - \hat{z}_0\|^2$
22:        end if
23:        
24:        // 反向传播
25:        $\nabla_\theta \mathcal{L}$ ← backprop($\mathcal{L}$)
26:        update($\theta$, $\nabla_\theta \mathcal{L}$)
27:    end for
28: end for
29: 
30: return $f_\theta$
```

**去噪器 $f_\theta$ 的详细过程**：

```
function f_\theta(z_t, t):
    // 输入投影
    $h = \text{Linear}_{in}(z_t)$  // $h \in \mathbb{R}^{B \times L^3 \times d_h}$
    
    // 时间嵌入
    $t_{emb} = \text{SinusoidalEmbedding}(t)$  // $t_{emb} \in \mathbb{R}^{B \times d_{emb}}$
    $t_{emb} = \text{expand}(t_{emb})$  // 广播到 $[B \times L^3 \times d_h]$
    
    // 构建图（网格锚点之间）
    $\mathcal{E} = \text{radius\_graph}(\mathcal{G}, r)$
    
    // EGNN去噪层
    for $l = 1$ to $L_{denoise}$ do
        for $(i,j) \in \mathcal{E}$ do
            $d_{ij} = \|g_i - g_j\|$
            $m_{ij} = \text{MLP}_e([h_i, h_j, d_{ij}, t_{emb,i}])$
        end for
        
        for each node $i$ do
            $m_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} m_{ij}$
            $h_i^{(l)} = h_i^{(l-1)} + \text{LayerNorm}(\text{MLP}_h([h_i^{(l-1)}, m_i]))$
        end for
    end for
    
    // 输出投影
    $\hat{z} = \text{Linear}_{out}(h^{(L_{denoise})})$
    return $\hat{z}$
end function
```

---

## Algorithm 3: DDPM采样生成

**输入**：训练好的去噪器 $f_\theta$，解码器 $D_\psi$，噪声调度参数

**输出**：生成的分子结构 $(X_{gen}, T_{gen})$

```
1: // 初始化：从标准高斯分布采样
2: $z_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$  // $z_T \in \mathbb{R}^{L^3 \times d}$
3: 
4: // 反向扩散过程
5: for $t = T$ down to $1$ do
6:     // 预测
7:     if method == "predict_epsilon" then
8:         $\hat{\varepsilon}_t = f_\theta(z_t, t)$
9:         // 从预测的噪声推导 $z_0$
10:        $\hat{z}_0 = \frac{z_t - \sqrt{1 - \bar{\alpha}_t} \hat{\varepsilon}_t}{\sqrt{\bar{\alpha}_t}}$
11:        // 计算 $z_{t-1}$ 的均值
12:        $\mu_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\varepsilon}_t \right)$
13:    else if method == "predict_x0" then
14:        $\hat{z}_0 = f_\theta(z_t, t)$
15:        // 从预测的 $z_0$ 推导噪声
16:        $\hat{\varepsilon}_t = \frac{z_t - \sqrt{\bar{\alpha}_t} \hat{z}_0}{\sqrt{1 - \bar{\alpha}_t}}$
17:        // 计算 $z_{t-1}$ 的均值
18:        $\mu_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( z_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \hat{\varepsilon}_t \right)$
19:    end if
20:    
21:    // 计算后验方差
22:    $\sigma_{t-1}^2 = \frac{\beta_t (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}$
23:    
24:    // 采样 $z_{t-1}$
25:    if $t > 1$ then
26:        $z_{t-1} = \mu_{t-1} + \sigma_{t-1} \cdot \varepsilon$, where $\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
27:    else
28:        $z_0 = \mu_0$  // 最后一步，不添加噪声
29:    end if
30: end for
31: 
32: // 从潜在码解码为矢量场
33: $Q \sim \mathcal{U}(\mathcal{B})$  // 采样查询点
34: $V = D_\psi(Q, z_0)$  // $V \in \mathbb{R}^{n_q \times K \times 3}$
35: 
36: // 从矢量场重建分子（梯度上升）
37: $(X_{gen}, T_{gen}) = \text{FieldToMolecule}(V, Q)$
38: 
39: return $(X_{gen}, T_{gen})$
```

---

## Algorithm 4: 从矢量场重建分子（梯度上升）

**输入**：矢量场 $V \in \mathbb{R}^{n_q \times K \times 3}$，查询点 $Q \in \mathbb{R}^{n_q \times 3}$，解码器 $D_\psi$，潜在码 $z$

**输出**：原子坐标 $X \in \mathbb{R}^{n \times 3}$ 和原子类型 $T \in \{0,\ldots,K-1\}^n$

```
1: // 为每个原子类型初始化采样点
2: for $k = 0$ to $K-1$ do
3:     $P^{(k)} \sim \mathcal{U}(\mathcal{B})$  // 随机初始化 $n_{points}$ 个点
4: end for
5: 
6: // 迭代优化
7: for $iter = 1$ to $n_{iter}$ do
8:     for each atom type $k$ do
9:         // 计算当前点的矢量场
10:        $V^{(k)} = D_\psi(P^{(k)}, z)$  // $V^{(k)} \in \mathbb{R}^{n_{points} \times K \times 3}$
11:        
12:        // 提取对应原子类型的梯度
13:        $\mathbf{g}^{(k)} = V^{(k)}[:, k, :]$  // $[n_{points} \times 3]$
14:        
15:        // 梯度上升
16:        $P^{(k)} \leftarrow P^{(k)} + \eta \cdot \mathbf{g}^{(k)}$
17:        
18:        // 检查收敛（可选）
19:        if $\|\mathbf{g}^{(k)}\|_{\text{mean}} < \tau$ and $iter > n_{min}$ then
20:            break
21:        end if
22:    end for
23: end for
24: 
25: // 聚类得到原子位置
26: for each atom type $k$ do
27:     $C^{(k)} = \text{DBSCAN}(P^{(k)}, \epsilon, n_{min})$  // 聚类
28:     $X^{(k)} = \{\text{centroid}(c) : c \in C^{(k)}\}$  // 取聚类中心
29:     $T^{(k)} = k \cdot \mathbf{1}_{|X^{(k)}|}$  // 分配原子类型
30: end for
31: 
32: // 合并所有原子类型
33: $X = \text{concat}(X^{(0)}, \ldots, X^{(K-1)})$
34: $T = \text{concat}(T^{(0)}, \ldots, T^{(K-1)})$
35: 
36: return $(X, T)$
```

---

## Algorithm 5: 分子到矢量场转换（GNF Converter）

**输入**：原子坐标 $X \in \mathbb{R}^{n \times 3}$，原子类型 $T \in \{0,\ldots,K-1\}^n$，查询点 $Q \in \mathbb{R}^{m \times 3}$

**输出**：矢量场 $V \in \mathbb{R}^{m \times K \times 3}$

```
1: for each query point $q_i \in Q$ do
2:     for each atom type $k \in \{0, \ldots, K-1\}$ do
3:         // 找到类型为 $k$ 的所有原子
4:         $\mathcal{A}_k = \{j : T_j = k\}$
5:         
6:         // 计算距离和权重（以Softmax Field为例）
7:         for each atom $j \in \mathcal{A}_k$ do
8:             $d_{ij} = \|q_i - X_j\|$
9:             $w_{ij} = \exp(-d_{ij}^2 / (2\sigma_k^2))$
10:        end for
11:        
12:        // 归一化权重
13:        $w_{ij} = \frac{w_{ij}}{\sum_{j' \in \mathcal{A}_k} w_{ij'}}$  // Softmax
14:        
15:        // 计算矢量场（指向原子的单位向量加权和）
16:        $\mathbf{v}_{ik} = \sum_{j \in \mathcal{A}_k} w_{ij} \cdot \frac{X_j - q_i}{\|X_j - q_i\| + \epsilon}$
17:    end for
18: end for
19: 
20: return $V = \{\mathbf{v}_{ik}\}_{i,k}$
```

---

## 符号说明

- $X \in \mathbb{R}^{n \times 3}$：原子坐标矩阵
- $T \in \{0,\ldots,K-1\}^n$：原子类型向量
- $K$：原子类型数量（如QM9中 $K=5$：C, H, O, N, F）
- $L$：锚点网格大小（通常 $L=8$，得到 $L^3=512$ 个锚点）
- $d$：潜在码维度（通常 $d=128$ 或 $1024$）
- $z \in \mathbb{R}^{L^3 \times d}$：潜在码
- $Q \in \mathbb{R}^{m \times 3}$：查询点坐标
- $V \in \mathbb{R}^{m \times K \times 3}$：矢量场
- $\mathcal{G} = \{g_j\}_{j=1}^{L^3}$：锚点网格
- $\beta_t, \alpha_t, \bar{\alpha}_t$：DDPM噪声调度参数
- $T$：扩散时间步数（通常 $T=100$ 或 $1000$）
- $\mathcal{N}(\mu, \Sigma)$：多元高斯分布
- $\mathcal{U}(\mathcal{S})$：集合 $\mathcal{S}$ 上的均匀分布

---

## 关键设计说明

1. **等变性保证**：EGNN层的坐标更新使用标量系数乘以方向向量，自动满足E(n)等变性。

2. **局部性**：通过k-NN和cutoff函数保证空间局部性，提高计算效率。

3. **多原子类型**：为每种原子类型独立建模矢量场，支持多类型分子生成。

4. **潜在空间生成**：在低维潜在空间（$L^3 \times d$）而非高维原子坐标空间生成，提高效率。

5. **连续场表示**：使用连续矢量场而非离散图，支持任意分辨率查询和重建。






