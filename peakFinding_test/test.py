#!/usr/bin/env python
# coding: utf-8


from ROOT import TFile, TTree
from array import array
import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='description')
parser.add_argument('-in', '--input', type=str, default='dataset_for_test.txt')
parser.add_argument('-out', '--output', type=str, default='prob.root')
parser.add_argument('-m', '--model', type=str, default='peak_finding.pth')
parser.add_argument('-n', '--nsize', type=int, default=-1)
parser.add_argument('--cut', type=float, default=0.95)
parser.add_argument('--dir', '-d', type=str, default='test')
#args = parser.parse_args(args=[])
args = parser.parse_args()

print(args)
filename_input = args.input
filename_output = args.output
nevt = args.nsize
my_thr = args.cut
dirName = args.dir
modelName = args.model

nevt = None if nevt < 0 else nevt
#dataset = pd.read_csv(filename_input, skipinitialspace=True)
dataset = pd.read_csv(filename_input, skipinitialspace=True, dtype={"EvtNo": np.int32, "ID": np.int32, "Shift": np.float32, "Sigma": np.int32, "Time": np.int32, "Time0": np.float32, "Time1": np.float32, "Time2": np.float32, "Time3": np.float32, "Time4": np.float32, "Time5": np.float32, "Time6": np.float32, "Time7": np.float32, "Time8": np.float32, "Time9": np.float32, "Time10": np.float32, "Time11": np.float32, "Time12": np.float32, "Time13": np.float32, "Time14": np.float32}) 
print('Dataset loaded ...')

ntime = 15
ndim = 1

dataset_label = dataset.loc[:, 'ID'].values

dataset_feature = dataset.loc[:, 'Time0':'Time%d' % (ntime-1)].values
dataset_feature = dataset_feature.reshape(dataset_feature.shape[0], ntime, ndim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_dataset = TensorDataset(torch.tensor(dataset_feature, dtype=torch.float32).to(device), torch.tensor(dataset_label, dtype=torch.float32).to(device))

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

model = Model().to(device)
model.load_state_dict(torch.load(modelName))
print('Model loaded ...')

def test(loader):
    model.eval()
    predicted_prob = []
    for data in loader:
        with torch.no_grad():
            outputs = model(data[0]) # Probability
            predicted_prob.append([outputs.item()])
    
    return np.array(predicted_prob)

predicted_prob=test(test_loader)

fileOut = TFile.Open(filename_output, "recreate")
treeOut = TTree("tmva", "tmva")
prob_rnn = array('d', [-999])
treeOut.Branch("prob_rnn", prob_rnn, "prob_rnn/D")

for prob in predicted_prob:
    prob_rnn[0] = prob
    treeOut.Fill()

fileOut.WriteTObject(treeOut)
fileOut.Close()

import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

bins = np.arange(-0.025, 1.075, 0.05)
#plt.figure(figsize=(15, 6))
_, _, bars = plt.hist(predicted_prob, bins, histtype='step', label="Predicted Probability")
n_prob1 = np.count_nonzero(predicted_prob==1.0)
n_probm = np.count_nonzero(predicted_prob>my_thr)
n_prob0 = np.count_nonzero(predicted_prob<0.01)
plt.xlabel("Probability")
plt.annotate('N(Total) = {}\nN(Prob=1.0) = {}\nN(Prob<0.01) = {}'.format(len(predicted_prob), n_prob1, n_prob0), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=16, horizontalalignment="center", verticalalignment='top')
#plt.show()
plt.savefig(f"{dirName}/ProbabilityAll.pdf",bbox_inches='tight')
plt.clf()

predicted_prob_sig = predicted_prob[dataset_label==1]
predicted_prob_bkg = predicted_prob[dataset_label==0]

_, _, bars = plt.hist(predicted_prob_sig, bins, histtype='step', label="PredictedProbability (for Signal)")
n_prob1_sig = np.count_nonzero(predicted_prob_sig==1.0)
n_probm_sig = np.count_nonzero(predicted_prob_sig>my_thr)
n_prob0_sig = np.count_nonzero(predicted_prob_sig<0.01)
plt.xlabel("Predicted Probability (for Signal)")
plt.annotate('N(Signal) = {}'.format(len(predicted_prob_sig)), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=16, horizontalalignment="center", verticalalignment='top')
#plt.show()
plt.savefig(f"{dirName}/ProbabilityAll_sig.pdf",bbox_inches='tight')
plt.clf()

_, _, bars = plt.hist(predicted_prob_bkg, bins, histtype='step', color="r", label="Predicted Probability (for Backgrouds)")
plt.xlabel("Predicted Probability (for Backgrouds)")
n_prob1_bkg = np.count_nonzero(predicted_prob_bkg==1.0)
n_probm_bkg = np.count_nonzero(predicted_prob_bkg>my_thr)
n_prob0_bkg = np.count_nonzero(predicted_prob_bkg<0.01)
plt.annotate('N(Backgrouds) = {}'.format(len(predicted_prob_bkg)), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=16, horizontalalignment="center", verticalalignment='top')
plt.savefig(f"{dirName}/ProbabilityAll_bkg.pdf",bbox_inches='tight')
plt.clf()

print('ALL:\nN_Total = {}\nN_Prob_1.0 = {}\nN_Prob_{} = {}\nN_Prob_0.01 = {}'.format(len(predicted_prob), n_prob1, my_thr, n_probm, n_prob0))
print('SIG:\nN_Signal = {}\nN_Prob_1.0 = {}\nN_Prob_{} = {}\nN_Prob_0.01 = {}'.format(len(predicted_prob_sig),n_prob1_sig, my_thr, n_probm_sig, n_prob0_sig))
print('BKG:\nN_Backgrouds = {}\nN_Prob_1.0 = {}\nN_Prob_{} = {}\nN_Prob_0.01 = {}'.format(len(predicted_prob_bkg), n_prob1_bkg, my_thr, n_probm_bkg, n_prob0_bkg))

print("Threshold =", my_thr)

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(dataset_label,predicted_prob,pos_label=1)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    f1 = 2 * (precisions * recalls)/(precisions + recalls)
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "r-", label="Recall")
    plt.plot(thresholds, f1[:-1], "g-.", label="F1")
    plt.xlabel("Threshold")
    plt.legend(loc="lower right")
    plt.ylim([0, 1])
    plt.vlines(my_thr, 0, 1.0, color='orange')
plot_precision_recall_vs_threshold(precision, recall, thresholds)

_, id  = find_nearest(thresholds,my_thr)
pt_pr = (id,thresholds[id],precision[id],recall[id])
print("id,thr,pre,rec =",pt_pr)

plt.savefig(f"{dirName}/TPRFcurve.pdf",bbox_inches='tight')
plt.clf()

plot_precision_recall_vs_threshold(precision, recall, thresholds)
#plt.figure(figsize=(15, 6))
plt.xlim([0.8,1.0])
plt.savefig(f"{dirName}/TPRFcurve_zoom.pdf",bbox_inches='tight')
plt.clf()

from sklearn.metrics import PrecisionRecallDisplay
pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.plot(pt_pr[3], pt_pr[2], "o")
plt.savefig(f"{dirName}/PRcurve.pdf",bbox_inches='tight')
plt.clf()

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
fpr, tpr, thr = roc_curve(dataset_label, predicted_prob, pos_label=1)

def plot_precision_recall_vs_threshold(fpr, tpr, thresholds):
    #f1 = 2 * (precisions * recalls)/(precisions + recalls)
    plt.plot(thresholds[1:], fpr[1:], "b--", label="False Positive Rate")
    plt.plot(thresholds[1:], tpr[1:], "r-", label="True Positive Rate")
    plt.xlabel("Threshold")
    plt.legend(loc="center right")
    plt.vlines(my_thr, 0, 1.0, color='orange')
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(fpr, tpr, thr)

_, id = find_nearest(thr,my_thr)
pt = (id,thr[id],fpr[id],tpr[id])
print("id,thr,fpr,tpr =",pt)

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(dataset_label, predicted_prob)
plt.plot(pt[2], pt[3], "o")
plt.savefig(f"{dirName}/ROCcurve.pdf",bbox_inches='tight')
plt.clf()

from sklearn.metrics import det_curve
from sklearn.metrics import DetCurveDisplay

fpr_det, fnr_det, thr_det =  det_curve(dataset_label, predicted_prob, pos_label=1)
for i in range(len(thr_det)):
    if abs(thr_det[i] - my_thr ) < 0.00001:
        print("id,thr,fpr_det,fnr_det =",i,thr_det[i],fpr_det[i],fnr_det[i])
        pt_det = (i,thr_det[i],fpr_det[i],fnr_det[i])
        break
DetCurveDisplay.from_predictions(dataset_label, predicted_prob)
plt.savefig(f"{dirName}/DETcurve.pdf",bbox_inches='tight')
plt.clf()

