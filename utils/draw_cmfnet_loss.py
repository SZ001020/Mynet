import matplotlib.pyplot as plt
import numpy as np
import os

def draw_from_npy(file_path):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    
    mean_losses = np.load(file_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_losses, label="Train Loss (Smooth)")
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('CMFNet Training Loss', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig('loss_curve_npy.png')
    plt.show()
    print("Loss curve saved as loss_curve_npy.png")

def draw_from_log(log_path):
    if not os.path.exists(log_path):
        print(f"Log file {log_path} does not exist. Please save your training output to this file.")
        return

    fig_loss = []
    with open(log_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Loss:' in line:
                try:
                    # 'Train (epoch 1/100) [0/1000 (0%)]\tLoss: 0.123456\tAccuracy: ...'
                    parts = line.split('Loss:')
                    loss_val = float(parts[1].split()[0])
                    fig_loss.append(loss_val)
                except:
                    continue

    if not fig_loss:
        print("No loss data found in log file.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(fig_loss, label="Train Loss")
    plt.xlabel('Steps (x100)', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('CMFNet Training Loss from Log', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig('loss_curve_log.png')
    plt.show()
    print("Loss curve saved as loss_curve_log.png")

# 使用方式：
# 如果你有 mean_losses.npy:
# draw_from_npy('/root/SSRS/CMFNet/mean_losses.npy')

# 如果你把终端日志存成了 train.log:
draw_from_log('/root/SSRS/CMFNet/train.log')
