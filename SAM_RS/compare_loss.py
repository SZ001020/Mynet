import re
import matplotlib.pyplot as plt

def parse_log(log_path):
    epochs = []
    total_loss = []
    accuracies = []

    # Pattern to match the training iteration lines
    pattern = re.compile(r"Train \(epoch (\d+)/\d+\) \[(\d+)/\d+ .*?Loss: ([\d\.]+)\s+Accuracy: ([\d\.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                l_total = float(match.group(3))
                acc = float(match.group(4))

                # Smooth the x-axis by adding iteration fraction to epoch
                epochs.append(epoch + iteration/1000.0)
                total_loss.append(l_total)
                accuracies.append(acc)
    
    return epochs, total_loss, accuracies

def plot_comparison(seg_path, sam_path, output_path):
    # Parse SEG data
    epochs_seg, loss_seg, acc_seg = parse_log(seg_path)
    # Parse SEG+BDY+OBJ data
    epochs_sam, loss_sam, acc_sam = parse_log(sam_path)

    if not epochs_seg or not epochs_sam:
        print("Error: Could not parse one of the log files.")
        return

    plt.figure(figsize=(14, 6))

    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs_seg, loss_seg, label='SEG Only', color='blue', alpha=0.6)
    plt.plot(epochs_sam, loss_sam, label='SEG+BDY+OBJ', color='red', alpha=0.6)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs_seg, acc_seg, label='SEG Only', color='blue', alpha=0.6)
    plt.plot(epochs_sam, acc_sam, label='SEG+BDY+OBJ', color='red', alpha=0.6)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    seg_log = '/root/SSRS/SAM_RS/SEG.log'
    sam_log = '/root/SSRS/SAM_RS/SEG+BDY+OBJ.log'
    output = '/root/SSRS/SAM_RS/Comparison_Loss_Curve.png'
    plot_comparison(seg_log, sam_log, output)
