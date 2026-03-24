import matplotlib.pyplot as plt
import re
import os

def plot_loss(log_path, output_png):
    iterations = []
    losses = []
    accuracies = []
    
    # 匹配模式: Train (epoch 1/50) [100/1000 (10%)]    Loss: 0.656265  Accuracy: 82.305908203125
    # 注意: 日志中可能有制表符或多个空格
    pattern = re.compile(r"Loss:\s+([\d.]+)\s+Accuracy:\s+([\d.]+)")
    
    count = 0
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                acc = float(match.group(2))
                losses.append(loss)
                accuracies.append(acc)
                iterations.append(count * 100) # 因为代码中是每 100 次迭代记录一次
                count += 1

    if not losses:
        print("No loss data found in log file.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iterations, losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个坐标轴绘制 Accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color)
    ax2.plot(iterations, accuracies, color=color, label='Train Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Training Loss and Accuracy\n({os.path.basename(log_path)})')
    fig.tight_layout()
    
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    import glob
    # 自动获取最新的 log 文件
    log_files = glob.glob('/root/SSRS/FTransUNet/train_*.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        plot_loss(latest_log, '/root/SSRS/FTransUNet/loss_plot.png')
    else:
        print("No log files found.")
