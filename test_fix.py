import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import numpy as np
import matplotlib.pyplot as plt


def compute_ece(probs, labels, num_bins=15):
    """Expected Calibration Error"""
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    confidences, predictions = probs.max(dim=-1)  # Shape: [N]
    labels = labels.view(-1)  # Ensure labels are 1D with shape [N]

    assert predictions.size(0) == labels.size(0), \
        f"Size mismatch: predictions {predictions.size()} vs labels {labels.size()}"

    accuracies = predictions.eq(labels)  # Shape: [N]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * bin_size / probs.size(0)
    return ece


def compute_normalized_entropy(probs):
    """Compute normalized entropy for each sample."""
    # Number of classes
    n_classes = probs.size(1)
    # Compute entropy
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)  # Use log base 2
    # Normalize by maximum entropy log2(n_classes)
    max_entropy = torch.log2(torch.tensor(n_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy



def plot_uncertainty(scores, title="Uncertainty Distribution", save_path="uncertainty_distribution.png"):
    """Plot uncertainty distribution"""
#     plt.hist(scores, bins=50, alpha=0.75)
    plt.hist(scores, bins=100, range=(0, 4.6), alpha=0.75) 
    plt.title(title)
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)  # 保存圖像
    plt.close()  # 關閉圖像避免資源佔用
#     plt.show()


def plot_ece(probs, labels, num_bins=15, save_path="reliability_diagram.png"):
    """Plot Expected Calibration Error"""
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum().item()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].mean().item()
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_sizes.append(bin_size)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_sizes.append(0)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_confs, bin_accs, marker='o', label="Reliability", color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.bar(bin_centers, np.array(bin_sizes) / sum(bin_sizes), width=0.05, alpha=0.5, color="orange", label="Sample Count")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.savefig(save_path)  # 保存圖像
    plt.close()  # 關閉圖像避免資源佔用
#     plt.show()

def plot_normalized_entropy(scores, title="Normalized Entropy Distribution", save_path="normalized_entropy_distribution.png"):
    """Plot normalized entropy distribution."""
    plt.hist(scores, bins=5, range=(0, 1), alpha=0.75)
    plt.title(title)
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Frequency")
    plt.savefig(save_path)  # 保存圖像
    plt.close()  # 關閉圖像避免資源佔用

def main(config):
    logger = config.get_logger('test')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=12
    )

    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    entropy_scores = []
    confidence_scores = []
    all_probs = []
    all_targets = []


    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)

            logits = model(data)
            if isinstance(logits, dict):  # Handle dictionary output
                logits = logits.get('logits', None)
                if logits is None:
                    raise ValueError("Model output does not contain the expected key 'logits'.")

            if len(logits.shape) == 4:  # Shape: [batch_size, num_classes, height, width]
                logits = logits.mean(dim=[2, 3])  # Average over spatial dimensions
            elif len(logits.shape) == 3:  # Shape: [batch_size, num_heads, num_classes]
                logits = logits.mean(dim=1)  # Average over heads

            probs = torch.softmax(logits, dim=-1)

            assert len(probs.shape) == 2, f"probs shape is incorrect: {probs.shape}"
            assert probs.size(0) == target.size(0), f"probs and target size mismatch: {probs.size(0)} vs {target.size(0)}"

            # Compute entropy and confidence
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            confidence = probs.max(dim=-1).values

            # Compute normalized entropy
            normalized_entropy = compute_normalized_entropy(probs)

            # Store results
            entropy_scores.extend(entropy.cpu().numpy())
            confidence_scores.extend(confidence.cpu().numpy())
            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())

            # Optionally store normalized entropy
            normalized_entropy_scores = normalized_entropy.cpu().numpy()
            
            
    all_probs = torch.cat(all_probs, dim=0)  # Shape: [N, C]
    all_targets = torch.cat(all_targets, dim=0)  # Shape: [N]

    assert len(all_probs.shape) == 2, f"all_probs shape is incorrect: {all_probs.shape}"
    assert len(all_targets.shape) == 1, f"all_targets shape is incorrect: {all_targets.shape}"
    assert all_probs.size(0) == all_targets.size(0), \
        f"Mismatch between all_probs and all_targets sizes: {all_probs.size(0)} vs {all_targets.size(0)}"

    ece_score = compute_ece(all_probs, all_targets)
    logger.info(f"Expected Calibration Error (ECE): {ece_score:.4f}")

    # Plot metrics and save images
    plot_uncertainty(entropy_scores, title="Entropy Distribution", save_path="entropy_distribution.png")
    plot_ece(all_probs, all_targets, save_path="reliability_diagram.png")
    plot_normalized_entropy(normalized_entropy_scores, title="Normalized Entropy Distribution", save_path="normalized_entropy_distribution.png")



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-l', '--log-config', default='logger/logger_config.json', type=str,
                      help='logging config file path (default: logger/logger_config.json)')
    args.add_argument('--use-wandb', action='store_true', help='Enable logging to Weights & Biases')
    args.add_argument("--validate", action='store_true', help='Run validation step')
    args.add_argument("--store-data", action='store_true', help='Store data for analysis')

    config, args = ConfigParser.from_args(args)
    main(config)
