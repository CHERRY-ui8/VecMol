#!/usr/bin/env python3
import csv

# 读取CSV文件
csv_file = 'exps/neural_field/nf_qm9/20251121/nf_evaluation_results_12.csv'
results = []

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # 检查所有CHONF真实数量是否都不为0
        gt_C = int(row['gt_C_count'])
        gt_H = int(row['gt_H_count'])
        gt_O = int(row['gt_O_count'])
        gt_N = int(row['gt_N_count'])
        gt_F = int(row['gt_F_count'])
        
        if gt_C > 0 and gt_H > 0 and gt_O > 0 and gt_N > 0 and gt_F > 0:
            results.append({
                'sample_idx': row['sample_idx'],
                'gt_C': gt_C,
                'gt_H': gt_H,
                'gt_O': gt_O,
                'gt_N': gt_N,
                'gt_F': gt_F
            })

# 输出结果
print(f"找到 {len(results)} 个分子的所有CHONF真实数量都不为0")
print("\n这些分子的sample_idx:")
if results:
    for r in results:
        print(f"sample_idx={r['sample_idx']}: C={r['gt_C']}, H={r['gt_H']}, O={r['gt_O']}, N={r['gt_N']}, F={r['gt_F']}")
else:
    print("没有找到符合条件的分子")
