from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from model import *
import uuid

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--wd1', type=float, default=0.01, help='weight decay (L2 loss on parameters).')
parser.add_argument('--wd2', type=float, default=5e-4, help='weight decay (L2 loss on parameters).')
parser.add_argument('--layer', type=int, default=64, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dataset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lambda.')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_citation(args.data)
cudaid = "cuda:" + str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
checkpt_file = 'pretrained/' + uuid.uuid4().hex + '.pt'
print(cudaid, checkpt_file)

model = GCNII(nfeat=features.shape[1],
              nlayers=args.layer,
              nhidden=args.hidden,
              nclass=int(labels.max()) + 1,
              dropout=args.dropout,
              lamda=args.lamda,
              alpha=args.alpha,
              variant=args.variant).to(device)

optimizer = optim.Adam([
    {'params': model.params1, 'weight_decay': args.wd1},
    {'params': model.params2, 'weight_decay': args.wd2},
], lr=args.lr)

# Function for adversarial example generation using FGSM
def generate_adversarial_example(model, features, adj, labels, idx_train, epsilon=0.01):
    model.train()
    optimizer.zero_grad()

    # Forward pass on the original data
    output = model(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss.backward()

    # Generate adversarial example using FGSM
    adversarial_features = features + epsilon * features.grad.sign()
    adversarial_features = Variable(adversarial_features, requires_grad=True)

    return adversarial_features

# Modify the train() function to include adversarial training
def train():
    model.train()
    optimizer.zero_grad()

    # Original data
    output = model(features, adj)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))

    # Adversarial training
    adversarial_features = generate_adversarial_example(model, features, adj, labels, idx_train)
    adversarial_output = model(adversarial_features, adj)
    adversarial_loss = F.nll_loss(adversarial_output[idx_train], labels[idx_train].to(device))

    # Combined loss
    total_loss = loss_train + adversarial_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item(), acc_train.item()

t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
for epoch in range(args.epochs):
    loss_tra, acc_tra = train()
    loss_val, acc_val = validate()
    if (epoch + 1) % 1 == 0:
        print('Epoch:{:04d}'.format(epoch + 1),
              'train',
              'loss:{:.3f}'.format(loss_tra),
              'acc:{:.2f}'.format(acc_tra * 100),
              '| val',
              'loss:{:.3f}'.format(loss_val),
              'acc:{:.2f}'.format(acc_val * 100))
    if loss_val < best:
        best = loss_val
        best_epoch = epoch
        acc = acc_val
        torch.save(model.state_dict(), checkpt_file)
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

if args.test:
    acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
print('Load {}th epoch'.format(best_epoch))
print("Test" if args.test else "Val", "acc.:{:.1f}".format(acc * 100))
