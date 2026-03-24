import re
import numpy as np

def calculate_real_metrics(log_path):
    with open(log_path, 'r') as f:
        content = f.read()
    
    # 提取混淆矩阵
    cm_match = re.search(r'Confusion matrix :\n\[\[(.*?)]]', content, re.DOTALL)
    if not cm_match:
        print("未在日志中找到混淆矩阵")
        return

    # 解析矩阵数据
    rows = cm_match.group(1).split(']\n [')
    matrix = []
    for row in rows:
        matrix.append([int(x) for x in row.split()])
    
    cm = np.array(matrix)
    
    # 提取前 6 类 (排除类别 6)
    real_cm = cm[:6, :6]
    
    diag = np.diag(real_cm)
    total_valid_pixels = np.sum(real_cm)
    correct_pixels = np.sum(diag)
    
    real_oa = (correct_pixels / total_valid_pixels) * 100
    
    # 计算各类的 F1
    f1_scores = []
    labels = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"]
    for i in range(6):
        precision = real_cm[i, i] / np.sum(real_cm[:, i]) if np.sum(real_cm[:, i]) > 0 else 0
        recall = real_cm[i, i] / np.sum(real_cm[i, :]) if np.sum(real_cm[i, :]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
    mean_f1 = np.mean(f1_scores[:5]) # 按照原始代码逻辑取前5类均值
    
    # 计算 MIoU
    intersection = diag
    union = np.sum(real_cm, axis=1) + np.sum(real_cm, axis=0) - diag
    miou_list = intersection / union
    mean_miou = np.mean(miou_list[:5])

    # 生成新日志内容
    new_log = f"""--- 修正后的真实指标 (已排除类别 6 Undefined 像素) ---
目标日志: {log_path}
有效像素数: {total_valid_pixels}
真实总体准确率 (Total Accuracy): {real_oa:.2f}%

各类别评价:
"""
    for i in range(6):
        new_log += f"{labels[i]:<10}: F1={f1_scores[i]:.4f}, IoU={miou_list[i]:.4f}\n"
    
    new_log += f"\n平均指标 (前 5 类均值):\n"
    new_log += f"Mean F1 Score: {mean_f1:.4f}\n"
    new_log += f"Mean mIoU    : {mean_miou:.4f}\n"
    new_log += "------------------------------------------------------"
    
    with open('real_metrics_summary.log', 'w') as f:
        f.write(new_log)
    
    print(new_log)

if __name__ == "__main__":
    calculate_real_metrics('/root/SSRS/ASMFNet/train_20260128_183934.log')
