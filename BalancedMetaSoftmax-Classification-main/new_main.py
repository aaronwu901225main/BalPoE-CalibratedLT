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
    for batch in data['test']:
        inputs = batch[0]  # 第一部分是 inputs
        targets = batch[1]  # 第二部分是 targets

        logits = training_model.forward(inputs.cuda())  # 計算模型輸出
        outputs.append(logits.cpu().detach().numpy())
        labels.append(targets.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 計算 ECE 和 MCE
    ece = ece_loss.loss(outputs, labels, n_bins=15, logits=True)
    mce = mce_loss.loss(outputs, labels, n_bins=15, logits=True)
    print(f"Test ECE: {ece}")
    print(f"Test MCE: {mce}")


print('ALL COMPLETED.')
