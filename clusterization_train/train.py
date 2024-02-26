# -*- coding: utf-8 -*-

import torch
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_add_pool

parser = argparse.ArgumentParser(description='description')
parser.add_argument('-dsp' ,'--datasetFilePeak', type=str, default='dataset/100k_peaks.txt')
parser.add_argument('-dst' ,'--datasetFileTime', type=str, default='dataset/100k_times.txt')
parser.add_argument('-out', '--output', type=str, default='cls.pth')
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('--ncov1', type=int, default=32)
parser.add_argument('--ncov2', type=int, default=32)
parser.add_argument('--ncov3', type=int, default=64)
parser.add_argument('--nmlp1', type=int, default=256)
parser.add_argument('--nmlp2', type=int, default=256)
parser.add_argument('--nmlp3', type=int, default=256)
parser.add_argument('--aggr', type=str, default='max')
parser.add_argument('--step', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('-k', '--k', type=int, default=4)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-n', '--number', type=str, default=datetime.now().strftime("%y%m%d_%H%M%S"))
parser.add_argument('-d', '--dir', type=str, default='cls_train_results')
#parser.add_argument('-norm', '--normalization', type=str, default="False")
#args = parser.parse_args(args=[])
args = parser.parse_args()

print(args)

peakfile = args.datasetFilePeak
timefile = args.datasetFileTime
number = args.number
ep = args.epochs
BATCHSIZE = args.batch_size
K = args.k
n_conv1 = args.ncov1
n_conv2 = args.ncov2
n_conv3 = args.ncov3
AGGR = args.aggr #'max'
n_mlp1 = args.nmlp1
n_mlp2 = args.nmlp2
n_mlp3 = args.nmlp3
LR = args.learning_rate
DIR = args.dir
STEP = args.step
GAMMA = args.gamma

# BATCHSIZE = 16
# n_conv1 = 32
# n_conv2 = 32
# n_conv3 = 64
# K = 4
# AGGR = 'max'
# n_mlp1 = 256
# n_mlp2 = 256
# n_mlp3 = 256
# #optimizer: RMSprop
# LR = 0.001

peaks, times = [], []
with open(timefile, "r") as f:
    for line in f:
        timearray = list(map(int, line.split()))
        times.append(timearray)
with open(peakfile, "r") as f:
    for line in f:
        peakarray = list(map(int, line.split()))
        peaks.append(peakarray)


peaks = [[0 if element == 2 else element for element in sublist] for sublist in peaks]
n_total = [len(sublist) for sublist in peaks]

print(len(peaks))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = []
for i in range(len(peaks)):
    if len(peaks[i]) == 1: continue
    x = torch.tensor(times[i], dtype=torch.float)
    x = torch.unsqueeze(x, dim=1)
    x = x.to(device)
    y = torch.tensor(peaks[i], dtype=torch.float)
    y = y.to(device)
    n = torch.tensor(n_total[i], dtype=torch.float)
    n = n.to(device)

    data = Data(x=x, y=y, n=n)
    dataset.append(data)

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=True)
print(len(train_loader),len(test_loader))

class Net(torch.nn.Module):
    def __init__(self, out_channels=2, k=K, aggr=AGGR):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 1      , n_conv1, n_conv1]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * n_conv1, n_conv2, n_conv2]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * n_conv2, n_conv3, n_conv3]), k, aggr)
        self.mlp = MLP([n_conv1+n_conv2+n_conv3, n_mlp1, n_mlp2, n_mlp3, out_channels], dropout=0.5, norm=None)

    def forward(self, data):
        x, batch = data.x, data.batch
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))

        return F.log_softmax(out, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(2, k=K, aggr=AGGR).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step=STEP, gamma=GAMMA)

def train():
    model.train()

    total_loss = 0
    correct = 0
    total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y.long())
        loss.backward()
        total_loss += loss.item() * data.num_graphs

        correct += out.max(dim=1)[1].eq(data.y).sum().item()
        total += data.num_nodes

        optimizer.step()
    return total_loss / train_dataset.__len__(), correct/total


def test(loader):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out.max(dim=1)[1]
            loss = F.nll_loss(out, data.y.long())

        total_loss += loss.item() * data.num_graphs

        correct += pred.eq(data.y).sum().item()
        total += data.y.shape[0]
    return total_loss / test_dataset.__len__(), correct / total

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(1,ep+1):
    train_loss, train_acc = train()
    val_loss, val_acc = test(test_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    if epoch % 2 == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'val_loss': val_loss,
                }, f'{DIR}/ckpt/ckpt{epoch}_loss{train_loss:.7f}-{val_loss:.7f}_acc{train_acc:4f}-{val_acc:4f}.pt')

    print(f'Epoch {epoch:03d}, Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc.: {val_acc:.4f}')
    scheduler.step()

torch.save(model.state_dict(), f'{DIR}/cls_model_weights_{number}.pth')
torch.save(model, f'{DIR}/cls_model_full_{number}.pth')

print(train_losses)
print(train_accs)
print(val_losses)
print(val_accs)

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Test Loss")
plt.legend()
# plt.show()
plt.savefig(f'{DIR}/loss.pdf')
plt.clf()

plt.plot(train_accs, label="Train Accuracy")
plt.plot(val_accs, label="Test Accuracy")
plt.legend()
# plt.show()
plt.savefig(f'{DIR}/acc.pdf')

