# 归一化逻辑移除总结

## 概述
本次更新完全移除了项目中的归一化逻辑，将基于resolution的格点设置改为基于真实空间距离的设置。主要变化包括：

## 主要更改

### 1. 配置文件更新
- **funcmol/configs/dset/qm9.yaml**: 移除`resolution: 0.25`，添加`grid_spacing: 2.0`
- **funcmol/configs/dset/drugs.yaml**: 移除`resolution: 0.25`，添加`grid_spacing: 2.0`
- **funcmol/configs/dset/cremp.yaml**: 移除`resolution: 0.25`，添加`grid_spacing: 2.0`
- **funcmol/configs/train_fm_*.yaml**: 移除`normalize_codes: 1`参数

### 2. 数据集处理逻辑更新
- **funcmol/dataset/dataset_field.py**:
  - 移除`scale_factor`计算：`1 / (resolution * grid_dim / 2)`
  - 将`resolution`参数改为`grid_spacing`
  - 更新`increments`计算，直接使用`grid_spacing`
  - 移除`_scale_molecule`和`_scale_batch_molecules`中的缩放逻辑
  - 更新`create_field_loaders`函数参数

### 3. 解码器更新
- **funcmol/models/decoder.py**:
  - 移除`_normalize_coords`和`_unnormalize_coords`函数
  - 更新`codes_to_molecules`方法，移除`unnormalize`参数
  - 更新`_refine_coords`方法，移除坐标反归一化逻辑
  - 移除`unnormalize_code`方法
  - 更新`get_grid`函数，使用真实空间距离

### 4. 工具函数更新
- **funcmol/utils/utils_nf.py**:
  - 移除`normalize_code`函数
  - 更新`infer_codes`和`infer_codes_occs_batch`，移除代码归一化
  - 移除`_normalize_coords`导入

- **funcmol/utils/utils_fm.py**:
  - 更新`compute_codes`和`compute_code_stats_offline`函数
  - 更新`process_codes`函数，移除归一化逻辑

### 5. 训练脚本更新
- **funcmol/train_fm.py**:
  - 移除`normalize_code`导入
  - 更新训练和验证循环中的归一化逻辑
  - 移除`compute_codes`和`compute_code_stats_offline`中的归一化参数

### 6. 评估脚本更新
- **funcmol/eval_nf.py**:
  - 移除`codes_to_molecules`调用中的`unnormalize`参数

### 7. 模型文件更新
- **funcmol/models/funcmol.py**:
  - 移除`codes_to_molecules`调用中的`unnormalize`参数

### 8. 调试文件更新
- **debug.py**:
  - 更新`FieldDataset`调用，使用`grid_spacing`替代`resolution`

## 参数变化

### 旧参数 → 新参数
- `resolution: 0.25` → `grid_spacing: 2.0`
- `normalize_codes: 1` → 移除（不再需要）
- `scale_factor` → 移除（不再需要）

### 格点设置变化
- **之前**: 每0.25单位一个格点（归一化空间）
- **现在**: 每2.0埃一个格点（真实空间）

### 坐标系统变化
- **之前**: 归一化坐标系统，需要缩放和反缩放
- **现在**: 真实坐标系统，直接使用埃单位

## 影响分析

### 正面影响
1. **简化逻辑**: 移除了复杂的归一化/反归一化流程
2. **物理意义**: 使用真实空间距离，更符合物理直觉
3. **调试友好**: 坐标值直接对应真实距离，便于调试
4. **参数透明**: 格点间距直接对应物理距离

### 需要注意的事项
1. **模型兼容性**: 现有训练好的模型可能需要重新训练
2. **参数调整**: 可能需要调整一些超参数以适应新的坐标系统
3. **性能影响**: 移除归一化可能影响数值稳定性，需要监控

## 测试建议
1. 运行单元测试确保基本功能正常
2. 在小数据集上测试训练流程
3. 验证生成的分子坐标是否合理
4. 检查梯度场计算是否正确

## 后续工作
1. 更新文档和注释
2. 调整超参数以适应新的坐标系统
3. 重新训练模型
4. 验证生成质量 