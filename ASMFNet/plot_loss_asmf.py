import matplotlib.pyplot as plt
import re
import os

def plot_loss(log_path, output_path):
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found.")
        return

    epochs = []
    batches = []
    losses = []
    global_iterations = []

    # Regular expression to catch: Train (epoch 1/100) [100/1000 (10%)]	Loss: 1.539907
    # Use \s+ to handle mixed spaces and tabs
    pattern = re.compile(r'Train \(epoch (\d+)/\d+\)\s+\[(\d+)/\d+.*Loss:\s+([\d\.]+)')

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                loss = float(match.group(3))
                
                epochs.append(epoch)
                batches.append(batch)
                losses.append(loss)
                
                # Assuming ~1000 batches per epoch based on the log [idx/1000]
                global_iterations.append((epoch - 1) * 1000 + batch)

    if not losses:
        print("No loss data found in the log file.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(global_iterations, losses, color='tab:red', linewidth=1.5, label='Training Loss')
    
    # Optional: Plot a smoothed version
    if len(losses) > 10:
        smoothed_losses = [sum(losses[max(0, i-5):i+5]) / len(losses[max(0, i-5):i+5]) for i in range(len(losses))]
        plt.plot(global_iterations, smoothed_losses, color='blue', alpha=0.5, linestyle='--', label='Smoothed Loss (Moving Avg)')

    plt.title('ASMFNet Training Loss Curve', fontsize=16)
    plt.xlabel('Global Iterations', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success: Loss plot saved to {output_path}")

if __name__ == "__main__":
    log_file = '/root/SSRS/ASMFNet/train_20260128_183934.log'
    save_file = '/root/SSRS/ASMFNet/asmfnet_loss_curve.png'
    plot_loss(log_file, save_file)
