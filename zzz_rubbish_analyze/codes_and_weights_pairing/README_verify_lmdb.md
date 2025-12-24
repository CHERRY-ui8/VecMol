# LMDB 对应关系验证工具

## 概述

`verify_lmdb_correspondence.py` 用于验证 `codes.lmdb` 和 `position_weights.lmdb` 之间是否按照顺序一一对应。

## 使用方法

### 基本验证（推荐）

随机抽取几个样本进行验证：

```bash
python verify_lmdb_correspondence.py \
    --codes_dir /path/to/codes \
    --split train \
    --num_samples 10 \
    --random_sample
```

### 顺序验证

按顺序验证前 N 个样本：

```bash
python verify_lmdb_correspondence.py \
    --codes_dir /path/to/codes \
    --split train \
    --num_samples 20 \
    --no-random_sample
```

### 完整验证（较慢但更彻底）

验证所有 keys 是否都能在两个数据库中找到：

```bash
python verify_lmdb_correspondence.py \
    --codes_dir /path/to/codes \
    --split train \
    --num_samples 10 \
    --verify_all
```

## 验证内容

脚本会检查以下内容：

1. **文件存在性检查**
   - `codes.lmdb` 和 `codes_keys.pt`
   - `position_weights.lmdb` 和 `position_weights_keys.pt`

2. **Keys 数量验证**
   - 两个 keys 文件的条目数量是否一致

3. **Keys 内容验证**
   - 两个 keys 文件中的每个 key 是否完全一致

4. **数据库条目数量验证**
   - 两个 LMDB 数据库中的条目数量是否一致
   - 数据库条目数量是否与 keys 数量一致

5. **样本对应关系验证**
   - 随机或顺序抽取样本
   - 验证每个样本的 key 是否能在两个数据库中找到
   - 显示每个样本的基本信息（shape, dtype, 统计信息）

6. **完整验证（可选）**
   - 验证所有 keys 是否都能在两个数据库中找到

## 输出示例

```
============================================================
验证 LMDB 对应关系: train
============================================================

文件检查:
  codes.lmdb: ✅ 存在
  codes_keys.pt: ✅ 存在
  position_weights.lmdb: ✅ 存在
  position_weights_keys.pt: ✅ 存在

加载 keys 文件...
  codes_keys: 10000 个条目
  position_weights_keys: 10000 个条目
✅ keys 数量一致: 10000

验证 keys 内容...
✅ 所有 keys 内容一致

打开 LMDB 数据库...

验证数据库条目数量...
  codes.lmdb: 10000 个条目
  position_weights.lmdb: 10000 个条目
✅ 数据库条目数量一致: 10000

采样验证 (10 个样本)...
  随机采样索引: [123, 456, 789, ...]
  索引 123 (key='123'):
    codes: shape=torch.Size([32, 32, 32, 128]), dtype=torch.float32, min=-2.3456, max=2.1234, mean=0.0123
    weights: shape=torch.Size([32, 32, 32]), dtype=torch.float32, min=0.0000, max=1.0000, mean=0.5000
    ✅ 两个数据库都能找到对应的数据
  ...

============================================================
✅ 验证通过: codes.lmdb 和 position_weights.lmdb 按照顺序一一对应
```

## 其他验证方法

### 方法 1: 检查 keys 文件

最简单的方法是直接比较两个 keys 文件：

```python
import torch

codes_keys = torch.load("codes_keys.pt", weights_only=False)
weights_keys = torch.load("position_weights_keys.pt", weights_only=False)

print(f"Codes keys: {len(codes_keys)}")
print(f"Weights keys: {len(weights_keys)}")
print(f"Keys match: {codes_keys == weights_keys}")
```

### 方法 2: 使用 Python 脚本快速验证

创建一个简单的验证脚本：

```python
import torch
import lmdb
import pickle

# 加载 keys
codes_keys = torch.load("codes_keys.pt", weights_only=False)
weights_keys = torch.load("position_weights_keys.pt", weights_only=False)

# 检查数量
assert len(codes_keys) == len(weights_keys), "Keys count mismatch!"

# 检查内容
for i, (ck, wk) in enumerate(zip(codes_keys, weights_keys)):
    assert str(ck) == str(wk), f"Key mismatch at index {i}: {ck} vs {wk}"

print(f"✅ Keys verification passed: {len(codes_keys)} keys match")

# 打开数据库验证
codes_db = lmdb.open("codes.lmdb", readonly=True)
weights_db = lmdb.open("position_weights.lmdb", readonly=True)

# 随机抽取几个样本验证
import random
sample_indices = random.sample(range(len(codes_keys)), min(10, len(codes_keys)))

for idx in sample_indices:
    key = str(codes_keys[idx]).encode('utf-8')
    
    with codes_db.begin() as txn:
        codes_value = txn.get(key)
        assert codes_value is not None, f"Missing key {idx} in codes.lmdb"
    
    with weights_db.begin() as txn:
        weights_value = txn.get(key)
        assert weights_value is not None, f"Missing key {idx} in position_weights.lmdb"

print(f"✅ Sample verification passed: {len(sample_indices)} samples checked")

codes_db.close()
weights_db.close()
```

### 方法 3: 检查转换日志

如果数据是通过 `convert_codes_to_lmdb.py` 转换的，可以检查转换日志：

- 两个数据库的条目数量应该一致
- 转换时使用的 `global_index` 应该相同

## 常见问题

### Q: 如果验证失败怎么办？

A: 如果验证失败，可能的原因：
1. 转换过程中出现错误
2. 两个数据库使用了不同的转换流程
3. 数据文件本身不匹配

解决方法：
- 重新转换数据：`python funcmol/dataset/convert_codes_to_lmdb.py --codes_dir <path> --splits train val test`
- 检查原始数据文件是否匹配

### Q: 验证需要多长时间？

A: 
- 基本验证（10个样本）：几秒钟
- 完整验证（所有keys）：取决于数据量，通常几分钟到几十分钟

### Q: 可以验证多个 split 吗？

A: 可以，分别运行：
```bash
python verify_lmdb_correspondence.py --codes_dir <path> --split train
python verify_lmdb_correspondence.py --codes_dir <path> --split val
python verify_lmdb_correspondence.py --codes_dir <path> --split test
```

## 注意事项

1. 确保 LMDB 文件没有被其他进程锁定
2. 验证过程是只读的，不会修改数据
3. 对于大型数据集，建议先进行采样验证，确认无误后再进行完整验证
