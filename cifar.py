from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

import os
import argparse
from resnet import ResNet18
from utils import *
from sklearn.manifold import TSNE



os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--bs', default=512, type=int, help='batch size')
parser.add_argument('--es', default=100, type=int, help='epoch size')
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay")
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def main():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = ResNet18(num_classes=10)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    for epoch in range(0, args.es):
        print('\nEpoch: %d   Learning rate: %f' % (epoch, optimizer.param_groups[0]['lr']))
        train_features,train_labels = train( optimizer, net, trainloader, criterion)
        visualization(train_features, train_labels)
        test_features, test_labels = test(net,testloader,criterion)
        visualization(test_features, test_labels)
        scheduler.step()


# Training
def train( optimizer, net, trainloader, criterion):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_features = []
    train_labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        features, outputs = net(inputs)
        train_features.append(features)
        train_labels.append(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_features,train_labels

def test(net,testloader,criterion):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_features = []
    test_labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            features, outputs = net(inputs)
            test_features.append(features)
            test_labels.append(targets)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return test_features, test_labels


def visualization(featureList, labelList):
    assert len(featureList) == len(labelList)
    assert len(featureList) > 0
    feature = featureList[0]
    label = labelList[0]
    for i in range(1, len(labelList)):
        feature = torch.cat([feature,featureList[i]],dim=0)
        label = torch.cat([label, labelList[i]], dim=0)
    feature =feature.cpu().detach().numpy()
    feature_embedded = TSNE(n_components=2).fit_transform(feature)
    print(f"feature shape: {feature.shape}")
    print(f"feature_embedded shape: {feature_embedded.shape}")
    print(f"label shape: {label.shape}")

if __name__ == '__main__':
    main()
