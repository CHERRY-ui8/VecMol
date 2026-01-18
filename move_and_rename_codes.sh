#!/bin/bash

# 临时脚本：移动和重命名codes文件
# 作用：
# 1. 移动 codes_001_batch_*.pt 和 position_weights_aug3_001_batch*.pt 到目标目录
# 2. 重命名 codes_001 -> codes_000, position_weights_aug3_001 -> position_weights_aug1_000

SOURCE_DIR="/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/codes_no_shuffle/train/temp_batches"
TARGET_DIR="/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/exps/neural_field/nf_drugs/20260113/lightning_logs/version_0/checkpoints/code_no_aug/train"

# 确保目标目录存在
mkdir -p "$TARGET_DIR"

echo "开始处理文件..."

# 处理 codes_001_batch_*.pt 文件
echo "处理 codes_001_batch_*.pt 文件..."
for file in "$SOURCE_DIR"/codes_001_batch_*.pt; do
    if [ -f "$file" ]; then
        # 提取批次号部分（例如：codes_001_batch_014920.pt -> 014920.pt）
        basename_file=$(basename "$file")
        # 将 codes_001_batch_ 替换为 codes_000_batch_
        new_name=$(echo "$basename_file" | sed 's/codes_001_batch_/codes_000_batch_/')
        target_path="$TARGET_DIR/$new_name"
        
        echo "移动: $basename_file -> $new_name"
        mv "$file" "$target_path"
    fi
done

# 处理 position_weights_aug3_001_batch*.pt 文件
echo "处理 position_weights_aug3_001_batch*.pt 文件..."
for file in "$SOURCE_DIR"/position_weights_aug3_001_batch*.pt; do
    if [ -f "$file" ]; then
        # 提取文件名
        basename_file=$(basename "$file")
        # 将 position_weights_aug3_001_batch_ 替换为 position_weights_aug1_000_batch_
        # 注意：用户写的是 aug1_00-_batch，这里按 aug1_000_batch 处理（保持格式一致）
        new_name=$(echo "$basename_file" | sed 's/position_weights_aug3_001_batch_/position_weights_aug1_000_batch_/')
        target_path="$TARGET_DIR/$new_name"
        
        echo "移动: $basename_file -> $new_name"
        mv "$file" "$target_path"
    fi
done

echo "完成！"
echo "已移动的文件数量："
echo "  codes_000_batch_*.pt: $(find "$TARGET_DIR" -name "codes_000_batch_*.pt" | wc -l)"
echo "  position_weights_aug1_000_batch*.pt: $(find "$TARGET_DIR" -name "position_weights_aug1_000_batch*.pt" | wc -l)"
