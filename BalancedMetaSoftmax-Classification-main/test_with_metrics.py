import torch
import numpy as np
import matplotlib.pyplot as plt


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

    confidences, predictions = probs.max(dim=-1)
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


def plot_Entropy(scores, title="Entropy Distribution", save_path="entropy_distribution.png"):
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


def load_probs_and_labels(model_path):
    """
    從 .pth 文件中加載機率分布和真實標籤。
    model_path: .pth 文件的路徑
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    probs = torch.tensor(checkpoint['probs'])  # 假設 .pth 包含 'probs' 鍵
    labels = torch.tensor(checkpoint['labels'])  # 假設 .pth 包含 'labels' 鍵
    return probs, labels


def evaluate_metrics_and_plots(probs, labels, num_bins=15):
    """
    計算 ECE 和 MCE，並繪製相關圖表。
    probs: 預測的機率分布 (Tensor)
    labels: 真實標籤 (Tensor)
    num_bins: 用於 ECE 和 MCE 計算的分箱數
    """
    # 計算 ECE 和 MCE
    ece = compute_ece(probs, labels, num_bins)
    mce = compute_mce(probs, labels, num_bins)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Mean Calibration Error (MCE): {mce:.4f}")

    # 計算熵和歸一化熵
    entropy = compute_entropy(probs)
    normalized_entropy = compute_normalized_entropy(probs, entropy)

    # 繪製圖表
    plot_Entropy(entropy, save_path="entropy_distribution.png")
    plot_normalized_entropy(normalized_entropy, save_path="normalized_entropy_distribution.png")
    plot_confidence_distribution(1 - normalized_entropy, save_path="confidence_distribution.png")
    plot_ece(probs, labels, num_bins, save_path="reliability_diagram.png")
    plot_mce(probs, labels, num_bins, save_path="mce_reliability_diagram.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=str, help="模型的 .pth 文件路徑，必須包含 'probs' 和 'labels'")
    parser.add_argument('--num_bins', default=15, type=int, help="分箱數，用於計算 ECE 和 MCE")
    args = parser.parse_args()

    # 加載機率分布和真實標籤
    probs, labels = load_probs_and_labels(args.model_path)

    # 計算指標並繪製圖表
    evaluate_metrics_and_plots(probs, labels, args.num_bins)
