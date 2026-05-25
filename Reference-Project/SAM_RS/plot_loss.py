import matplotlib.pyplot as plt
import re
import numpy as np

log_file = '/root/SSRS/SAM_RS/train_20260131_152327.log'
output_plot = '/root/SSRS/SAM_RS/loss_curve.png'

# Data storage
iterations = []
loss_ce = []
loss_bdy = []
loss_obj = []
loss_total = []

val_epochs = []
val_miou = []
val_f1 = []

# Regex to match the loss lines
# Example: Train (epoch 1/50) [0/1000 (0%)]	Loss_ce: 1.729886	Loss_boundary: 0.906115	Loss_object: 0.013931	Loss: 1.834428	Accuracy: 32.80029296875
pattern = re.compile(r"Train \(epoch (\d+)/\d+\) \[(\d+)/\d+ .*?\]\tLoss_ce: ([\d\.]+)\tLoss_boundary: ([\d\.]+)\tLoss_object: ([\d\.]+)\tLoss: ([\d\.]+)")
miou_pattern = re.compile(r"mean MIoU: ([\d\.]+)")
f1_pattern = re.compile(r"mean F1Score: ([\d\.]+)")

epoch_count = 0
with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            batch = int(match.group(2))
            global_it = (epoch - 1) * 1000 + batch
            
            iterations.append(global_it)
            loss_ce.append(float(match.group(3)))
            loss_bdy.append(float(match.group(4)))
            loss_obj.append(float(match.group(5)))
            loss_total.append(float(match.group(6)))
            epoch_count = max(epoch_count, epoch)
        
        miou_match = miou_pattern.search(line)
        if miou_match:
            val_miou.append(float(miou_match.group(1)))
            val_epochs.append(len(val_miou))
            
        f1_match = f1_pattern.search(line)
        if f1_match:
            val_f1.append(float(f1_match.group(1)))

if not iterations:
    print("No loss data found in log file.")
    exit()

# Smoothing the curves for better visualization
def smooth(y, box_pts):
    if len(y) < box_pts:
        return y
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.figure(figsize=(15, 12))

# Subplot 1: Loss
plt.subplot(2, 1, 1)
plt.plot(iterations, loss_total, alpha=0.2, color='gray', label='Total Loss (raw)')
plt.plot(iterations, smooth(loss_total, 20), color='black', linewidth=2, label='Total Loss (smooth)')
plt.plot(iterations, smooth(loss_ce, 20), label='CE Loss', alpha=0.8)
plt.plot(iterations, smooth(loss_bdy, 20), label='Boundary Loss', alpha=0.8)
plt.plot(iterations, smooth(loss_obj, 20), label='Object Loss', alpha=0.8)
plt.xlabel('Global Iteration', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.title('Training Loss Curves', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)

# Subplot 2: Validation Metrics
plt.subplot(2, 1, 2)
if val_miou:
    plt.plot(val_epochs, val_miou, marker='o', label='mean MIoU', color='blue')
if val_f1:
    plt.plot(val_epochs, val_f1, marker='x', label='mean F1Score', color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Validation Metrics', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, epoch_count + 1, max(1, epoch_count // 10)))

plt.tight_layout()
plt.savefig(output_plot)
print(f"Plot saved to {output_plot}")
