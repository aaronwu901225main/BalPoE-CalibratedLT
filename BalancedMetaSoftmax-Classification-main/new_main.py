"""
Copyright (c) Facebook, Inc. and its affiliates.  
All rights reserved.

This source code is licensed under the license found in the LICENSE file 
in the root directory of this source tree.
"""

import os
import argparse
import pprint
import matplotlib.pyplot as plt
import numpy as np
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import source_import, get_value
from scipy.special import softmax
from utils import ECELoss, MCELoss  # 修改此處，將之前提供的 CELoss 類別放入檔案中並匯入

data_root = {'ImageNet': './dataset/ImageNet',
             'Places': './dataset/Places-LT',
             'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
             'CIFAR10': './dataset/CIFAR10',
             'CIFAR100': './dataset/CIFAR100'}

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--model_dir', type=str, default=None)
parser.add_argument('--save_feat', type=str, default='')

# KNN testing parameters 
parser.add_argument('--knn', default=False, action='store_true')
parser.add_argument('--feat_type', type=str, default='cl2n')
parser.add_argument('--dist_type', type=str, default='l2')

# Learnable tau
parser.add_argument('--val_as_train', default=False, action='store_true')

args = parser.parse_args()

def update(config, args):
    # Update configuration with args
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = get_value(config['training_opt']['batch_size'], args.batch_size)
    return config

# Load configuration
with open(args.cfg, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
pprint.pprint(config)

def split2phase(split):
    if split == 'train' and args.val_as_train:
        return 'train_val'
    else:
        return split

# Initialize ECE and MCE calculators
ece_loss = ECELoss()
mce_loss = MCELoss()

if not test_mode:
    # Training phase
    sampler_defs = training_opt['sampler']
    sampler_dic = None
    splits = ['train', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=split2phase(x), 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    cifar_imb_ratio=training_opt.get('cifar_imb_ratio', None))
            for x in splits}

    training_model = model(config, data, test=False)
    training_model.train()

    # After training: calculate ECE and MCE on validation set
    outputs, labels = [], []
    for inputs, targets in data['val']:
        logits = training_model.forward(inputs.cuda())
        outputs.append(logits.cpu().detach().numpy())
        labels.append(targets.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    ece = ece_loss.loss(outputs, labels, n_bins=15, logits=True)
    mce = mce_loss.loss(outputs, labels, n_bins=15, logits=True)
    print(f"Validation ECE: {ece}")
    print(f"Validation MCE: {mce}")

    # Plot reliability diagram
    bin_lowers = ece_loss.bin_lowers
    bin_uppers = ece_loss.bin_uppers
    bin_centers = (bin_lowers + bin_uppers) / 2

    plt.figure(figsize=(8, 6))
    plt.bar(bin_centers, ece_loss.bin_acc, width=0.1, alpha=0.6, label="Accuracy")
    plt.plot(bin_centers, ece_loss.bin_conf, 'o-', label="Confidence", color='red')
    plt.plot([0, 1], [0, 1], '--', label="Perfect Calibration", color='gray')
    plt.title("Reliability Diagram")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

else:
    # Testing phase
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    print('Under testing phase.')

    splits = ['train', 'val', 'test']
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False,
                                    cifar_imb_ratio=training_opt.get('cifar_imb_ratio', None))
            for x in splits}

    training_model = model(config, data, test=True)
    training_model.load_model(args.model_dir)

    # Testing phase
    outputs, labels = [], []

    print(f"Type of training_model: {type(training_model)}")
    print(f"Attributes of training_model: {dir(training_model)}")

    # 遍歷測試數據集
    for batch in data['test']:
        inputs = batch[0]  # 獲取輸入
        targets = batch[1]  # 獲取標籤

        # 計算模型輸出（logits）
        logits = training_model.batch_forward(inputs.cuda())  # 嘗試 batch_forward 方法
        outputs.append(logits.cpu().detach().numpy())
        labels.append(targets.cpu().numpy())

    # 合併批次輸出
    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 計算模型機率
    import torch
    probs = torch.softmax(torch.tensor(outputs), dim=-1)

    # 計算準確率
    predictions = probs.argmax(dim=-1)
    accuracy = (predictions == torch.tensor(labels)).float().mean().item()

    # 計算熵
    def calculate_entropy(probs):
        return -np.sum(probs * np.log(probs + 1e-12), axis=1)

    entropies = calculate_entropy(probs.numpy())
    total_entropy = np.sum(entropies)
    average_entropy = np.mean(entropies)

    # 計算標準化熵
    normalized_entropies = entropies / np.log(probs.size(-1))
    total_normalized_entropy = np.sum(normalized_entropies)
    average_normalized_entropy = np.mean(normalized_entropies)

    # 計算 ECE 和 MCE
    ece = ece_loss.loss(outputs, labels, n_bins=15, logits=True)
    mce = mce_loss.loss(outputs, labels, n_bins=15, logits=True)

    # 過濾樣本的統計
    threshold = 0.8
    filtered = (probs.max(dim=-1).values > threshold)
    filtered_accuracy = (predictions[filtered] == torch.tensor(labels)[filtered]).float().mean().item()
    total_filtered_samples = filtered.sum().item()
    total_dropped_samples = len(labels) - total_filtered_samples
    drop_rate = total_dropped_samples / len(labels)

    # 打印結果
    print(f"Total test samples: {len(labels)}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Total Entropy: {total_entropy:.4f}")
    print(f"Average Entropy: {average_entropy:.4f}")
    print(f"Total normalized Entropy: {total_normalized_entropy:.4f}")
    print(f"Average normalized Entropy: {average_normalized_entropy:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Total filtered samples: {total_filtered_samples}")
    print(f"Total dropped samples: {total_dropped_samples}")
    print(f"Drop Rate: {drop_rate:.4f}")
    print(f"Filtered Accuracy: {filtered_accuracy:.4f}")
    print(f"Maximum Calibration Error (MCE): {mce:.4f}")

    # 繪製 ECE 和 MCE 圖
    def plot_ece(probs, labels, num_bins=15, save_path="ece_reliability_diagram.png"):
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
        plt.title("ECE Reliability Diagram")
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
        if max_error_bin is not None:
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

    plot_ece(probs, torch.tensor(labels), save_path="ece_reliability_diagram.png")
    plot_mce(probs, torch.tensor(labels), save_path="mce_reliability_diagram.png")

    print("Reliability diagrams saved as ece_reliability_diagram.png and mce_reliability_diagram.png.")
print('ALL COMPLETED.')
