"""Copyright (c) Facebook, Inc. and its affiliates. 
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Portions of the source code are from the OLTR project which
notice below and in LICENSE in the root directory of
this source tree.

Copyright (c) 2019, Zhongqi Miao
All rights reserved.
"""

import os
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import yaml
from utils import source_import, get_value
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定義數據根目錄

data_root = {
    'ImageNet': './dataset/ImageNet',
    'Places': './dataset/Places-LT',
    'iNaturalist18': '/checkpoint/bykang/iNaturalist18',
    'CIFAR10': './dataset/CIFAR10',
    'CIFAR100': './dataset/CIFAR100',
}

# 定義新函式

def compute_ece(probs, labels, num_bins=15):
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

def compute_entropy(probs):
    epsilon = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
    return entropy

def compute_normalized_entropy(probs, entropy):
    n_classes = probs.size(1)
    max_entropy = torch.log(torch.tensor(n_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    return normalized_entropy

def compute_mce(probs, labels, num_bins=15):
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

# 修改后的代码开始
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
    # Change parameters
    config['model_dir'] = get_value(config['model_dir'], args.model_dir)
    config['training_opt']['batch_size'] = \
        get_value(config['training_opt']['batch_size'], args.batch_size)

    # Testing with KNN
    if args.knn and args.test:
        training_opt = config['training_opt']
        classifier_param = {
            'feat_dim': training_opt['feature_dim'],
            'num_classes': training_opt['num_classes'], 
            'feat_type': args.feat_type,
            'dist_type': args.dist_type,
            'log_dir': training_opt['log_dir']}
        classifier = {
            'def_file': './models/KNNClassifier.py',
            'params': classifier_param,
            'optim_params': config['networks']['classifier']['optim_params']}
        config['networks']['classifier'] = classifier
    
    return config

# ============================================================================
# LOAD CONFIGURATIONS
with open(args.cfg, 'r') as f:  # 修改: 添加 'r' 模式
    config = yaml.load(f, Loader=yaml.SafeLoader)  # 修改: 添加 Loader=yaml.SafeLoader
config = update(config, args)

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits
training_opt = config['training_opt']
relatin_opt = config['memory']
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

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
        elif sampler_defs['type'] == 'MetaSampler':  # Add option for Meta Sampler
            learner = source_import(sampler_defs['def_file']).get_learner()(
                num_classes=training_opt['num_classes'],
                init_pow=sampler_defs.get('init_pow', 0.0),
                freq_path=sampler_defs.get('freq_path', None)
            ).cuda()
            sampler_dic = {
                'batch_sampler': True,
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'meta_learner': learner, 'batch_size': training_opt['batch_size']}
            }
    else:
        sampler_dic = None

    splits = ['train', 'train_plain', 'val']
    if dataset not in ['iNaturalist18', 'ImageNet']:
        splits.append('test')
    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=split2phase(x), 
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None)
            for x in splits}

    if sampler_defs and sampler_defs['type'] == 'MetaSampler':   # todo: use meta-sampler
        cbs_file = './data/ClassAwareSampler.py'
        cbs_sampler_dic = {
                'sampler': source_import(cbs_file).get_sampler(),
                'params': {'is_infinite': True}
            }
        # use Class Balanced Sampler to create meta set
        data['meta'] = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase='train' if 'CIFAR' in dataset else 'val',
                                    batch_size=sampler_defs.get('meta_batch_size', training_opt['batch_size'], ),
                                    sampler_dic=cbs_sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None,
                                    meta=True)
        training_model = model(config, data, test=False, meta_sample=True, learner=learner)
    else:
        training_model = model(config, data, test=False)

    training_model.train()

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data",
                            UserWarning)

    print('Under testing phase, we load training data simply to calculate \
           training data number for each class.')

    if 'iNaturalist' in training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    else:
        splits = ['train', 'val', 'test']
        test_split = 'test'
    if 'ImageNet' == training_opt['dataset']:
        splits = ['train', 'val']
        test_split = 'val'
    if args.knn or True:
        splits.append('train_plain')

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')],
                                    dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None, 
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False,
                                    cifar_imb_ratio=training_opt['cifar_imb_ratio'] if 'cifar_imb_ratio' in training_opt else None)
            for x in splits}
    
    training_model = model(config, data, test=True)
    # training_model.load_model()
    training_model.load_model(args.model_dir)
    if args.save_feat in ['train_plain', 'val', 'test']:
        saveit = True
        test_split = args.save_feat
    else:
        saveit = False
    
    training_model.eval(phase=test_split, openset=test_open, save_feat=saveit)
    
    if output_logits:
        training_model.output_logits(openset=test_open)

    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(data[test_split]):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[:2]  # 假設第一個是數據，第二個是標籤
            else:
                raise ValueError(f"Unexpected batch format: {batch}")
            inputs, targets = inputs.cuda(), targets.cuda()
            logits = training_model(inputs)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    # 計算指標
    ece = compute_ece(all_probs, all_targets)
    mce = compute_mce(all_probs, all_targets)

    print(f"ECE: {ece:.4f}")
    print(f"MCE: {mce:.4f}")

    # 繪製圖表
    plot_ece(all_probs, all_targets, save_path="reliability_diagram.png")
    plot_mce(all_probs, all_targets, save_path="mce_reliability_diagram.png")

print('ALL COMPLETED.')
