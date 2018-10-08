"""
Author: Saravanakumar Shanmugam Sakthivadivel, 2018
Github: https://github.com/codewithsk/graph-cnn.mxnet

The Ohio State University
Graph Convolutional Network

File: train.py
Description: Training script for graph convolutional network
"""
import time
import argparse
import numpy as np


import mxnet as mx
from mxnet import autograd, gluon

from utils import load_data, accuracy
from model import GCN

# Parse command line arguments
parser = argparse.ArgumentParser() # pylint: disable=invalid-name
parser.add_argument('--num-gpu', type=int, default=-1,
                    help='Select GPU to train on. -1 for CPU.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args() #pylint: disable=invalid-name

# Set up context.
ctx = None # pylint: disable=invalid-name

if args.num_gpu == -1:
    ctx = mx.cpu() # pylint: disable=invalid-name
else:
    ctx = mx.gpu(args.num_gpu) # pylint: disable=invalid-name

# Set seed for random number generators in numpy and mxnet
np.random.seed(args.seed)
mx.random.seed(args.seed)

adj, features, labels, idx_train, idx_val, idx_test = load_data() # pylint: disable=invalid-name

model = GCN(nfeat=features.shape[1], # pylint: disable=invalid-name
            nhid=args.hidden,
            nclass=int(labels.max().asnumpy().item()) + 1,
            dropout=args.dropout)

model.collect_params().initialize()
trainer = gluon.Trainer(model.collect_params(), # pylint: disable=invalid-name
                        'adam',
                        {'learning_rate': args.lr,})


# Note: Original implementation uses
# Negative Log Likelihood and not
# SoftmaxCrossEntropyLoss
loss = gluon.loss.SoftmaxCrossEntropyLoss() # pylint: disable=invalid-name

accs = [] # pylint: disable=invalid-name

for epoch in range(args.epochs):
    t = time.time()
    with autograd.record():
        output = model(features, adj)
        loss_train = loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        accs.extend(acc_train.asnumpy())
        loss_train.backward()

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(np.mean(loss_train.asnumpy())),
          'acc_train: {:.4f}'.format(acc_train.asnumpy().item()),
          'time: {:.4f}s'.format(time.time() - t))

    trainer.step(1)

print(
    'Training Accuracy: ', accuracy(output[idx_train], labels[idx_train]).asnumpy().item(), '\n',
    'Validation Accuracy: ', accuracy(output[idx_val], labels[idx_val]).asnumpy().item(), '\n',
    'Test Accuracy: ', accuracy(output[idx_test], labels[idx_test]).asnumpy().item()
)
