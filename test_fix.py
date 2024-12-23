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


def compute_entropy(probs):
    """計算每個樣本的熵。"""
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return entropy


def compute_normalized_entropy(probs, entropy):
    """計算每個樣本的標準化熵。"""
    n_classes = probs.size(1)
    max_entropy = torch.log(torch.tensor(n_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy


def compute_mce(probs, labels, num_bins=15):
    """
    計算模型的 Mean Calibration Error (MCE)。
    probs: Tensor, 機率分布
    labels: Tensor, 真實標籤
    num_bins: 分箱數
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


def plot_mce(probs, labels, num_bins=15, save_path="mce_reliability_diagram.png"):
    """繪製 MCE 可靠性圖並標記最大校準誤差。"""
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
    if max_error_bin:
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


def main(config):
    logger = config.get_logger('test')

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=256,
        shuffle=False,
        training=False,
        num_workers=12
    )
    total_samples = len(data_loader.dataset)
    model = config.init_obj('arch', module_arch)
    logger.info(f"Loading checkpoint: {config.resume}")
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint['state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    entropy_scores = []
    confidence_scores = []
    normalized_entropy_scores = []
    all_probs = []
    all_targets = []
    correct_predictions = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            probs = torch.softmax(logits, dim=-1)

            predictions = probs.argmax(dim=-1)
            correct_predictions += predictions.eq(target).sum().item()

            entropy = compute_entropy(probs)
            normalized_entropy = compute_normalized_entropy(probs, entropy)
            confidences = 1 - normalized_entropy

            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())
            entropy_scores.extend(entropy.cpu().numpy())
            normalized_entropy_scores.extend(normalized_entropy.cpu().numpy())
            confidence_scores.extend(confidences.cpu().numpy())

    test_accuracy = correct_predictions / total_samples
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    ece = compute_ece(torch.cat(all_probs), torch.cat(all_targets))
    logger.info(f"Expected Calibration Error (ECE): {ece:.4f}")
    mce = compute_mce(torch.cat(all_probs), torch.cat(all_targets))
    logger.info(f"Mean Calibration Error (MCE): {mce:.4f}")

    plot_mce(torch.cat(all_probs), torch.cat(all_targets), save_path="mce_reliability_diagram.png")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable')
    config, args = ConfigParser.from_args(args)
    main(config)
