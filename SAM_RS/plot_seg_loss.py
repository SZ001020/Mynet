import re
import matplotlib.pyplot as plt

def plot_loss(log_path, output_path):
    epochs = []
    iterations = []
    loss_ce = []
    loss_boundary = []
    loss_object = []
    total_loss = []
    accuracies = []

    # Regex to match the training lines
    # Train (epoch 1/50) [0/1000 (0%)]	Loss_ce: 1.832003	Loss_boundary: 0.912063	Loss_object: 0.017383	Loss: 1.832003	Accuracy: 6.4300537109375
    pattern = re.compile(r"Train \(epoch (\d+)/\d+\) \[(\d+)/\d+ .*?Loss_ce: ([\d\.]+)\s+Loss_boundary: ([\d\.]+)\s+Loss_object: ([\d\.]+)\s+Loss: ([\d\.]+)\s+Accuracy: ([\d\.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                l_ce = float(match.group(3))
                l_b = float(match.group(4))
                l_o = float(match.group(5))
                l_total = float(match.group(6))
                acc = float(match.group(7))

                epochs.append(epoch + iteration/1000.0)
                loss_ce.append(l_ce)
                loss_boundary.append(l_b)
                loss_object.append(l_o)
                total_loss.append(l_total)
                accuracies.append(acc)

    if not epochs:
        print("No data found in log file.")
        return

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, total_loss, label='Total Loss')
    plt.plot(epochs, loss_ce, label='CE Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_loss('/root/SSRS/SAM_RS/SEG.log', '/root/SSRS/SAM_RS/SEG_loss_curve.png')
