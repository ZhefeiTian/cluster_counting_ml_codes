#!/usr/bin/env python
# coding: utf-8

from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='description')
parser.add_argument('-in', '--input', type=str, default='weighed_dataset.txt')
parser.add_argument('-out', '--output', type=str, default='peak_finding.pt')
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
parser.add_argument('-n', '--number', type=str, default=str(int(time.time())))
#args = parser.parse_args(args=[])
args = parser.parse_args()

print(args)

filename = args.input
modelname = args.output
number = args.number
ep = args.epochs
lr = args.learning_rate
bs = args.batch_size

dataset = pd.read_csv(filename, skipinitialspace=True)

ntime = 15
ndim = 1

dataset_label = dataset.loc[:, 'ID'].values

dataset_feature = dataset.loc[:, 'Time0':'Time%d' % (ntime-1)]
dataset_feature = dataset_feature.values
dataset_feature = dataset_feature.reshape(dataset_feature.shape[0], ntime, ndim)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TensorDataset(torch.tensor(dataset_feature, dtype=torch.float32).to(device), torch.tensor(dataset_label, dtype=torch.float32).to(device))

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2])

train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

print(len(train_loader), len(val_loader))

class Model(nn.Module):
    def __init__(self, input_len=15, embedding=False):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=1, num_layers=1, hidden_size=32, batch_first=True)
        self.fc3 = torch.nn.Linear(32, 32)
        self.fc4 = torch.nn.Linear(32, 1)

    def forward(self, x):
        ula, (h, _) = self.lstm(x)
        out = h[-1]
        out = F.relu(self.fc3(out))
        clf = F.sigmoid(self.fc4(out))

        return clf


model = Model()
model = model.to(device)

criterion = nn.BCELoss()
# optimizer = optim.RMSprop(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_losses = []
train_accs = []
val_losses = []
val_accs = []


def train():
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        optimizer.zero_grad()
        outputs = model(data[0])
        pred = torch.where(outputs > 0.5, 1, 0)
        labels = data[1].unsqueeze(1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()

        correct += pred.eq(labels).sum().item()
        total += len(data[0])
        optimizer.step()

    return total_loss / train_dataset.__len__(), correct/total


def test(loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        with torch.no_grad():
            outputs = model(data[0])
            pred = torch.where(outputs > 0.5, 1, 0)
            labels = data[1].unsqueeze(1)
            loss = criterion(outputs, labels)
        total_loss += loss.item()

        correct += pred.eq(labels).sum().item()
        total += len(data[0])
    return total_loss / val_dataset.__len__(), correct / total

for epoch in range(1, ep+1):
    train_loss, train_acc = train()
    val_loss, val_acc = test(val_loader)
    
    if epoch % 2 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                }, f'train_results_{number}/ckpt{epoch}_loss{train_loss:.7f}-{val_loss:.7f}_acc{train_acc:4f}-{val_acc:4f}.pt')
    
    print(f"Epoch {epoch}: loss: {train_loss}, acc: {train_acc}; val loss: {val_loss}, val acc: {val_acc}")

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)


torch.save(model.state_dict(), "train_results_{}/peak_finding_{}.pth".format(number,number))
torch.save(model, 'train_results_{}/peak_finding_full_{}.pth'.format(number,number))

print(train_losses)
print(train_accs)
print(val_losses)
print(val_accs)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Test Loss")
plt.legend()
# plt.show()
plt.savefig("train_results_{}/loss.pdf".format(number))
plt.clf()

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Test Accuracy")
plt.legend()
#plt.show()
plt.savefig("train_results_{}/acc.pdf".format(number))

print("Done!")
