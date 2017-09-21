'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import os
import time
import argparse
import pandas as pd
import json
from collections import defaultdict

import keras
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np.squeeze(y_train, axis=1)
y_test = np.squeeze(y_test, axis=1)

train_datagen = ImageDataGenerator(featurewise_center=True, 
                                   featurewise_std_normalization=True)
train_datagen.fit(x_train)
test_datagen = ImageDataGenerator(featurewise_center=True, 
                                  featurewise_std_normalization=True)
test_datagen.fit(x_train)

from models import *
from utils import progress_bar
from torch.autograd import Variable
from numpy_dataset import NumpyDataset

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epochs', type=int, default=350, help='number of epochs to train')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--result', type=str,  default=randomhash, help='result path')
parser.add_argument('--no-padding', default=True, action='store_false', help='use padding')
args = parser.parse_args()

assert not(os.path.exists(args.result)), "result dir already exists!"
os.makedirs(args.result)
config_str = json.dumps(vars(args))
config_file = os.path.join(args.result, 'config')
config_file_object = open(config_file, 'w')
config_file_object.write(config_str)
config_file_object.close()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
pad = 4 if args.no_padding else 0
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=pad),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_batch_size = 128
num_train_batches = int(np.ceil(len(x_train)/train_batch_size))
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainset = NumpyDataset(x_train, np.squeeze(y_train, axis=1), transform=transform_train)
#trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)
trainloader = train_datagen.flow(x_train, y_train, batch_size=train_batch_size)

test_batch_size = 100
num_test_batches = int(np.ceil(len(x_test)/test_batch_size))
#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#testset = NumpyDataset(x_test, np.squeeze(y_test, axis=1), transform=transform_test)
#testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)
testloader = test_datagen.flow(x_test, y_test, batch_size=test_batch_size)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], gamma=0.1)

train_metrics = defaultdict(list)
test_metrics = defaultdict(list)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets).long()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)

        curr_total = targets.size(0)
        curr_correct = predicted.eq(targets.data).cpu().sum()
        total += curr_total
        correct += curr_correct

        train_metrics['cross_entropy'].append(loss.data[0])
        train_metrics['accuracy'].append(100.*curr_correct/curr_total)

        progress_bar(batch_idx, num_train_batches, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx == num_train_batches-1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets).long()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, num_test_batches, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if batch_idx == num_test_batches-1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break

    # Save checkpoint.
    acc = 100.*correct/total
    test_metrics['cross_entropy'].append(test_loss/(batch_idx+1))
    test_metrics['accuracy'].append(acc)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

for epoch in range(start_epoch, args.epochs):
    scheduler.step()
    train(epoch)
    test(epoch)

    pd_train_metrics = pd.DataFrame(train_metrics)
    pd_train_metrics.to_csv(os.path.join(args.result, 'train_metrics.csv'))
    pd_test_metrics = pd.DataFrame(test_metrics)
    pd_test_metrics.to_csv(os.path.join(args.result, 'test_metrics.csv'))

