import torch
import numpy as np
import matplotlib.pyplot as plt
from data import dataloader
from run_networks import model
from utils import source_import, get_value
import yaml
import argparse


def compute_entropy(probs):
    """
    計算每個樣本的熵。
    probs: Tensor, shape [N, C]，表示 N 個樣本的機率分布
    """
    epsilon = 1e-8  # 避免 log(0)
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return entropy


def compute_normalized_entropy(probs, entropy):
    """
    計算每個樣本的歸一化熵。
    probs: Tensor, shape [N, C]，表示 N 個樣本的機率分布
    entropy: Tensor，表示每個樣本的熵
    """
    n_classes = probs.size(1)
    max_entropy = torch.log(torch.tensor(n_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def compute_ece(probs, labels, num_bins=15):
    """
    計算模型的 Expected Calibration Error (ECE)。
    probs: Tensor, shape [N, C]，機率分布
    labels: Tensor, shape [N]，真實標籤
    num_bins: ECE 的分箱數
    """
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    entropy = compute_entropy(probs)
    normalized_entropy = compute_normalized_entropy(probs, entropy)
    confidences = 1 - normalized_entropy

    _, predictions = probs.max(dim=-1)
    labels = labels.view(-1)

    accuracies = predictions.eq(labels)
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * bin_size / probs.size(0)
    return ece


def compute_mce(probs, labels, num_bins=15):
    """
    計算模型的 Mean Calibration Error (MCE)。
    probs: Tensor, shape [N, C]，機率分布
    labels: Tensor, shape [N]，真實標籤
    num_bins: MCE 的分箱數
    """
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]

    confidences, predictions = probs.max(dim=-1)
    labels = labels.view(-1)

    accuracies = predictions.eq(labels)
    max_calibration_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            max_calibration_error = max(max_calibration_error, torch.abs(conf_in_bin - acc_in_bin).item())
    return max_calibration_error


def plot_Entropy(scores, title="Entropy Distribution", save_path="Entropy_distribution.png"):
    plt.hist(scores, bins=100, range=(0, 4.6), alpha=0.75)
    plt.title(title)
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


def plot_normalized_entropy(scores, title="Normalized Entropy Distribution", save_path="normalized_entropy_distribution.png"):
    plt.hist(scores, bins=100, range=(0, 1), alpha=0.75)
    plt.title(title)
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


def plot_confidence_distribution(confidence_scores, title="Confidence Distribution", save_path="confidence_distribution.png"):
    plt.hist(confidence_scores, bins=100, range=(0, 1), alpha=0.75)
    plt.title(title)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)
    plt.close()


def plot_ece(probs, labels, num_bins=15, save_path="reliability_diagram.png"):
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
    plt.savefig(save_path)
    plt.close()


def plot_mce(probs, labels, num_bins=15, save_path="mce_reliability_diagram.png"):
    bins = torch.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    bin_accs = []
    bin_confs = []
    max_error_bin = None
    max_calibration_error = 0

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum().item()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean().item()
            conf_in_bin = confidences[in_bin].mean().item()
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)

            calibration_error = abs(conf_in_bin - acc_in_bin)
            if calibration_error > max_calibration_error:
                max_calibration_error = calibration_error
                max_error_bin = (conf_in_bin, acc_in_bin)
        else:
            bin_accs.append(0)
            bin_confs.append(0)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_confs, bin_accs, marker='o', label="Reliability", color='blue')
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.scatter(max_error_bin[0], max_error_bin[1], color='red', label="Max Calibration Error", zorder=5)
    plt.annotate(f"Max Error: {max_calibration_error:.2f}",
                 xy=max_error_bin, xytext=(max_error_bin[0] + 0.1, max_error_bin[1] - 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, color='red')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("MCE Reliability Diagram")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def test_model_with_metrics(config, data, model_dir, num_bins=15):
    training_model = model(config, data, test=True)
    training_model.load_model(model_dir)

    probs, labels = training_model.evaluate_probs()
    labels = torch.tensor(labels)
    probs = torch.tensor(probs)

    ece = compute_ece(probs, labels, num_bins)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    mce = compute_mce(probs, labels, num_bins)
    print(f"Mean Calibration Error (MCE): {mce:.4f}")

    entropy = compute_entropy(probs)
    normalized_entropy = compute_normalized_entropy(probs, entropy)

    plot_Entropy(entropy, save_path="entropy_distribution.png")
    plot_normalized_entropy(normalized_entropy, save_path="normalized_entropy_distribution.png")
    plot_confidence_distribution(1 - normalized_entropy, save_path="confidence_distribution.png")
    plot_ece(probs, labels, num_bins, save_path="reliability_diagram.png")
    plot_mce(probs, labels, num_bins, save_path="mce_reliability_diagram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help="配置文件的路徑")
    parser.add_argument('--model_dir', required=True, type=str, help="模型保存目錄")
    parser.add_argument('--num_bins', default=15, type=int, help="分箱數，用於計算 ECE 和 MCE")
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    training_opt = config['training_opt']
    dataset = training_opt['dataset']
    data_root = {'CIFAR10': './dataset/CIFAR10', 'CIFAR100': './dataset/CIFAR100'}
    data = {split: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                        dataset=dataset, phase=split,
                                        batch_size=training_opt['batch_size'],
                                        num_workers=training_opt['num_workers'],
                                        shuffle=False)
            for split in ['train', 'val', 'test']}

    test_model_with_metrics(config, data, args.model_dir, args.num_bins)
