import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import numpy as np
import matplotlib.pyplot as plt


# def compute_ece(probs, labels, num_bins=15):
#     """Expected Calibration Error"""
#     bins = torch.linspace(0, 1, num_bins + 1)
#     bin_lowers = bins[:-1]
#     bin_uppers = bins[1:]

#     confidences, predictions = probs.max(dim=-1)  # Shape: [N]
#     labels = labels.view(-1)  # Ensure labels are 1D with shape [N]

#     assert predictions.size(0) == labels.size(0), \
#         f"Size mismatch: predictions {predictions.size()} vs labels {labels.size()}"

#     accuracies = predictions.eq(labels)  # Shape: [N]

#     ece = 0
#     for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
#         in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
#         bin_size = in_bin.float().sum()
#         if bin_size > 0:
#             acc_in_bin = accuracies[in_bin].float().mean()
#             conf_in_bin = confidences[in_bin].mean()
#             ece += torch.abs(conf_in_bin - acc_in_bin) * bin_size / probs.size(0)
#     return ece
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

    # 使用 normalized entropy 來計算新的置信度
    entropy = compute_entropy(probs)
    normalized_entropy = compute_normalized_entropy(probs, entropy)
    confidences = 1 - normalized_entropy  # 置信度越高，entropy 越低

    # 預測類別
    _, predictions = probs.max(dim=-1)
    labels = labels.view(-1)  # 確保 labels 是 1D

    # 準確性
    accuracies = predictions.eq(labels)  # [N]，布林值表示預測是否正確

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = in_bin.float().sum()
        if bin_size > 0:
            acc_in_bin = accuracies[in_bin].float().mean()  # 該分箱內的準確率
            conf_in_bin = confidences[in_bin].mean()        # 該分箱內的平均置信度
            ece += torch.abs(conf_in_bin - acc_in_bin) * bin_size / probs.size(0)
    return ece


def compute_entropy(probs):
    """
    計算每個樣本的熵。
    probs: Tensor, shape [N, C]，表示 N 個樣本的機率分布
    """
    epsilon = 1e-8  # 避免 log(0)
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return entropy



def compute_normalized_entropy(probs,entropy):
    """Compute normalized entropy for each sample."""
    # Number of classes
    n_classes = probs.size(1)
    max_entropy = torch.log(torch.tensor(n_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy



def plot_Entropy(scores, title="Entropy Distribution", save_path="Entropy_distribution.png"):
    """Plot uncertainty distribution"""
#     plt.hist(scores, bins=50, alpha=0.75)
    plt.hist(scores, bins=100, range=(0, 4.6), alpha=0.75) 
    plt.title(title)
    plt.xlabel("Uncertainty Score")
    plt.ylabel("Frequency")
    plt.savefig(save_path)  # 保存圖像
    plt.close()  # 關閉圖像避免資源佔用
#     plt.show()




def plot_normalized_entropy(scores, title="Normalized Entropy Distribution", save_path="normalized_entropy_distribution.png"):
    """Plot normalized entropy distribution."""
    plt.hist(scores, bins=100, range=(0, 1), alpha=0.75)
    plt.title(title)
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Frequency")
    plt.savefig(save_path)  # 保存圖像
    plt.close()  # 關閉圖像避免資源佔用




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

def plot_confidence_distribution(confidence_scores, title="Confidence Distribution", save_path="confidence_distribution.png"):
    """Plot confidence score distribution."""
    plt.hist(confidence_scores, bins=100, range=(0, 1), alpha=0.75)
    plt.title(title)
    plt.xlabel("Confidence Score")
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
    total_samples = len(data_loader.dataset)
    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
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
    correct_predictions = 0  # 用於累計正確預測數


    #threshold變數宣告
    threshold = 0  # 設定 confidence 閾值
    total_filtered_samples = 0
    total_dropped_samples = 0
    correct_predictions_filtered = 0

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)

            logits = model(data)
            if isinstance(logits, dict):
                logits = logits.get('logits', None)
                if logits is None:
                    raise ValueError("Model output does not contain the expected key 'logits'.")

            if len(logits.shape) == 4:
                logits = logits.mean(dim=[2, 3])
            elif len(logits.shape) == 3:
                logits = logits.mean(dim=1)

            probs = torch.softmax(logits, dim=-1)

            assert len(probs.shape) == 2, f"probs shape is incorrect: {probs.shape}"
            assert probs.size(0) == target.size(0), f"probs and target size mismatch: {probs.size(0)} vs {target.size(0)}"

            # Compute predictions and compare with targets
            predictions = probs.argmax(dim=-1)  # 每個樣本的預測類別
            correct_predictions += predictions.eq(target).sum().item()  # 累加正確預測數

            # Compute entropy and confidence
            entropy = compute_entropy(probs)

            # Compute normalized entropy
            normalized_entropy = compute_normalized_entropy(probs, entropy)
            confidences = 1 - normalized_entropy  # 置信度越高，entropy 越低



            #加上threshold的設定
            threshold = 0.8 # 設定 confidence 閾值
            valid_mask = confidences > threshold  # 布林遮罩，True 表示保留樣本

            # 過濾樣本
            filtered_probs = probs[valid_mask]
            filtered_targets = target[valid_mask]
            filtered_predictions = filtered_probs.argmax(dim=-1)
                        # 計算保留樣本的正確預測數
            correct_filtered = filtered_predictions.eq(filtered_targets).sum().item()

            # 統計結果
            num_filtered = valid_mask.sum().item()  # 保留樣本數量
            num_dropped = target.size(0) - num_filtered  # 放棄作答的樣本數量

            # 累積統計
            correct_predictions_filtered += correct_filtered  # 累加正確預測數
            total_filtered_samples += num_filtered  # 累加有效樣本數
            total_dropped_samples += num_dropped  # 累加放棄樣本數
            



            # Store results
            entropy_scores.extend(entropy.cpu().numpy())
#             confidence_scores.extend(confidence.cpu().numpy())
            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())
            normalized_entropy_scores.extend(normalized_entropy.cpu().numpy())
            confidence_scores.extend(confidences.cpu().numpy())
    # Calculate accuracy原本的accuracy計算
    test_accuracy = correct_predictions / total_samples
    
    
    # 放棄作答的比例
    drop_rate = total_dropped_samples / total_samples
    # 有效樣本的準確率
    filtered_accuracy = correct_predictions_filtered / total_filtered_samples if total_filtered_samples > 0 else 0.0

    # Log results
    logger.info(f"Total test samples: {total_samples}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Expected Calibration Error (ECE): {compute_ece(torch.cat(all_probs), torch.cat(all_targets)):.4f}")
    logger.info(f"Total Entropy: {sum(entropy_scores):.4f}")
    logger.info(f"Average Entropy: {np.mean(entropy_scores):.4f}")
    logger.info(f"Total normalized Entropy: {sum(normalized_entropy_scores):.4f}")
    logger.info(f"Average normalized Entropy: {np.mean(normalized_entropy_scores):.4f}")
    
    logger.info(f"threshold: {threshold}")    
    logger.info(f"Total filtered samples: {total_filtered_samples}")
    logger.info(f"Total dropped samples: {total_dropped_samples}")
    logger.info(f"Drop Rate: {drop_rate:.4f}")
    logger.info(f"Filtered Accuracy: {filtered_accuracy:.4f}")

    # Plot metrics and save images
    plot_Entropy(entropy_scores, title="Entropy Distribution", save_path="entropy_distribution.png")
    plot_ece(torch.cat(all_probs), torch.cat(all_targets), save_path="reliability_diagram.png")
    plot_normalized_entropy(normalized_entropy_scores, title="Normalized Entropy Distribution", save_path="normalized_entropy_distribution.png")
    plot_confidence_distribution(confidence_scores, title="Confidence Distribution", save_path="confidence_distribution.png")




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
