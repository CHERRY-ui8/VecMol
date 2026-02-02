# VecMol 环境配置指南

## 当前状态
- ✅ `env.yml` 文件已存在
- ✅ `setup.py` 文件已创建
- ❌ 系统未检测到 conda 或 mamba

## 步骤 1: 安装 Conda 或 Mamba

### 选项 A: 安装 Miniconda (推荐)
```bash
# 下载 Miniconda 安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 运行安装脚本
bash Miniconda3-latest-Linux-x86_64.sh

# 按照提示完成安装，然后重新加载 shell
source ~/.bashrc
```

### 选项 B: 安装 Mambaforge (更快，推荐用于科学计算)
```bash
# 下载 Mambaforge 安装脚本
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh

# 运行安装脚本
bash Mambaforge-Linux-x86_64.sh

# 按照提示完成安装，然后重新加载 shell
source ~/.bashrc
```

## 步骤 2: 创建环境

安装完成后，执行以下命令：

```bash
# 如果使用 mamba
mamba env create -f env.yml

# 或者如果使用 conda
conda env create -f env.yml
```

## 步骤 3: 激活环境

```bash
# 根据 env.yml，环境名称是 vecmol_oss
conda activate vecmol_oss

# 或者如果环境名称是 vecmol
conda activate vecmol
```

## 步骤 4: 安装包

```bash
# 确保在项目根目录
cd /home/huayuchen/Neurl-voxel

# 以开发模式安装
pip install -e .
```

## 验证安装

```bash
# 检查环境
conda env list

# 检查包是否安装
python -c "import vecmol; print('VecMol 安装成功！')"
```

## 注意事项

1. 如果 conda/mamba 已安装但命令不可用，可能需要初始化：
   ```bash
   # 对于 bash
   source ~/.bashrc
   
   # 或者手动初始化
   eval "$(/path/to/conda/bin/conda shell.bash hook)"
   ```

2. `env.yml` 中定义的环境名称是 `vecmol_oss`，但 README 中提到可能是 `vecmol`，请根据实际情况使用正确的名称。

3. 如果遇到权限问题，确保有写入权限。
