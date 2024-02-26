#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# @Author      : TIAN Zhefei (tianzhefei@whu.edu.cn)
# @Date        : 2023-04-20 08:09:55 CST
# @Modified    : 2023-04-20 08:09:55 CST
# @Version     : 1.0
# @Description : 
"""
A python script.
"""

# import sys
import torch
import ROOT
import numpy as np
from array import array
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dsp' ,'--datasetFilePeak', type=str, default='peaks.txt')
parser.add_argument('-dst' ,'--datasetFileTime', type=str, default='times.txt')
parser.add_argument('-sample', '--sampleFile', type=str, default='signal_10.root')
parser.add_argument('-i', '--input', type=str, default='counting_test.root')
parser.add_argument('-m', '--model', type=str, default='cls.pth')
parser.add_argument('-t', '--title', type=str, default='')
parser.add_argument('-p', '--particle', type=str, default='')
parser.add_argument('-d', '--dir', type=str, default='cls_test')
parser.add_argument('-thr', '--threshold', type=str, default='0.26')
parser.add_argument('--nsize', type=int, default=-1)
parser.add_argument('--isData', type=bool, default=False)
parser.add_argument('--useD2', type=bool, default=False)

#args = parser.parse_args([])
args = parser.parse_args()
print(args)

timeFile = args.datasetFileTime
peakFile = args.datasetFilePeak
sampleFile = args.sampleFile
pfFile = args.input
title = args.title
particle = args.particle
dirName = args.dir
modelName = args.model

n_conv1 = 32
n_conv2 = 32
n_conv3 = 64
K = 4
AGGR = 'max'
n_mlp1 = 256
n_mlp2 = 256
n_mlp3 = 256
LR = 0.0001

print(f'n_conv1 = {n_conv1}')
print(f'n_conv2 = {n_conv2}')
print(f'n_conv3 = {n_conv3}')
print(f'K = {K}')
print(f'AGGR = {AGGR}')
print(f'n_mlp1 = {n_mlp1}')
print(f'n_mlp2 = {n_mlp2}')
print(f'n_mlp3 = {n_mlp3}')
print(f'LR = {LR}')

plt.rcParams.update({'font.size': 18})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

prob_thr = torch.tensor([0.26])

peaks_for_test, times_for_test = [], []
with open(timeFile, "r") as f:
    for line in f:
        array_times = list(map(int, line.split()))
        times_for_test.append(array_times)
with open(peakFile, "r") as f:
    for line in f:
        array_peaks = list(map(int, line.split()))
        peaks_for_test.append(array_peaks)

peaks_for_test = [[0 if element == 2 else element for element in sublist] for sublist in peaks_for_test]
n_total_for_test = [len(sublist) for sublist in peaks_for_test]
ncls_for_test = [sum(sublist) for sublist in peaks_for_test]

import matplotlib.pyplot as plt
plt.hist(ncls_for_test, bins=(max(ncls_for_test)-min(ncls_for_test)))
plt.savefig(f"{dirName}/ncls.png")

dataset_for_test = []
for i in range(len(peaks_for_test)):
    if len(peaks_for_test[i]) == 1: continue
    x = torch.tensor(times_for_test[i], dtype=torch.float)
    x = torch.unsqueeze(x, dim=1)
    x = x.to(device)
    y = torch.tensor(peaks_for_test[i], dtype=torch.float)
    y = y.to(device)
    n = torch.tensor(n_total_for_test[i], dtype=torch.float)
    n = n.to(device)

    data = Data(x=x, y=y, n=n)
    dataset_for_test.append(data)

test_loader = DataLoader(dataset_for_test, batch_size=1, shuffle=False)

import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv, global_add_pool

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


import numpy as np
import matplotlib.pyplot as plt

model = Net().to(device)
model.load_state_dict(torch.load(modelName))
# model = torch.load(modelName)
print('Model loaded ...')


def draw_test(loader, thr):
    print("thr = {}".format(thr.item()))
    log_thr = torch.log(thr)
    model.eval()
    ncls_pred = []
    ncls_true = []
    ratios = []
    predicted_prob_tensor = []
    dataset_label_tensor = []
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
            out = out[:,1]
        predicted_prob_tensor.append(torch.exp(out))
        dataset_label_tensor.append(data.y)
        pred = torch.where(out >= log_thr, torch.ones_like(out), torch.zeros_like(out))
        n_pred = torch.sum(pred == 1).item()
        n_true = torch.sum(data.y == 1).item()
        n_total = data.n.item()

        ncls_pred.append(n_pred)
        ncls_true.append(n_true)
        ratios.append(n_true/n_total)

    predicted = np.array(ncls_pred)
    dataset_target = np.array(ncls_true)
    n_ratio = np.array(ratios)
    diff = predicted - dataset_target
    
    predicted_prob = torch.cat(predicted_prob_tensor, dim=0).cpu().numpy()
    dataset_label = torch.cat(dataset_label_tensor, dim=0).cpu().numpy()

    pre_min, pre_max = np.min(predicted), np.max(predicted)
    tar_min, tar_max = np.min(dataset_target), np.max(dataset_target)
    dif_min ,dif_max = np.min(diff), np.max(diff)

    pre_min = int(pre_min)-3
    pre_max = int(pre_max)+3
    tar_min = int(tar_min)-3
    tar_max = int(tar_max)+3

    mean_diff = np.mean(diff)
    mean_abs_diff = np.mean(np.abs(diff))
    print("mean_diff =",mean_diff)
    print("mean_abs_diff =",mean_abs_diff)
    print("diff_min =", dif_min)
    print("diff_max =", dif_max)

    np.savez(f"{dirName}/test_arrays_thr{thr.item()}.npz", n_ratio=n_ratio, predicted=predicted, dataset_target=dataset_target, diff=diff)

    # Generate root file
    print("Generate root file of results...")
    file = ROOT.TFile(sampleFile, "read")
    tree = file.Get("sim")
    file2 = ROOT.TFile(pfFile, "read")
    tree2 = file2.Get("sim")
    ncls = array('i', [-1])
    ncls_sampling = array('i', [-1])
    ncount = array('i', [-1])

    tree.SetBranchAddress('ncls', ncls,)
    tree2.SetBranchAddress('ncount', ncount)
    tree2.SetBranchAddress('ncls', ncls_sampling)

    n = tree.GetEntries()
    if n != tree2.GetEntries():
        raise ValueError("n != n_sampling")
    
    print("Root files have been read...")

    file_out = ROOT.TFile(f"{dirName}/result_{particle}_{title}_{thr.item()}.root", "recreate")
    tree_out = ROOT.TTree('sim', 'sim')

    nclusters = array('i', [-1])
    sampling_rate_out = array('d', [-1.])
    ncount_out = array('i', [-1])
    ncls_out = array('i', [-1])
    ncls_sampling_out = array('i', [-1])
    ncls_pf = array('i', [-1])
    
    tree_out.Branch('nclusters', nclusters, 'nclusters/I') # cls results
    tree_out.Branch('ncount', ncount_out, 'ncount/I') # pf results->n of peaks
    tree_out.Branch('ncls', ncls_out, 'ncls/I') # truth
    tree_out.Branch('ncls_sampling', ncls_sampling_out, 'ncls_sampling/I') # wf truth
    tree_out.Branch('ncls_pf', ncls_pf, 'ncls_pf/I') # pf results->ncls, target of cls

    for i in range(n):
        # wf_out.clear()
        tree.GetEntry(i)
        tree2.GetEntry(i)
        ncls_out[0] = ncls[0]
        ncls_sampling_out[0] = ncls_sampling[0]
        ncount_out[0] = ncount[0]
        nclusters[0] = predicted[i]
        ncls_pf[0] = dataset_target[i]

        tree_out.Fill()
    file_out.WriteTObject(tree_out)
    file_out.Close()
    print(f"{dirName}/result_{particle}_{title}_{thr.item()}.root generated.")
    
# TEST PLOT
    my_thr = thr.item()
    print("Threshold =", my_thr)
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx
    
    plt.clf()
    bins = np.arange(-0.025, 1.075, 0.05)
    _, _, bars = plt.hist(predicted_prob, bins, histtype='step', label="Predicted Probability")
    n_prob1 = np.count_nonzero(predicted_prob==1.0)
    n_probm = np.count_nonzero(predicted_prob>my_thr)
    n_prob0 = np.count_nonzero(predicted_prob<0.01)
    plt.xlabel("Probability")
    plt.annotate('N(Total) = {}\nN(Prob=1.0) = {}\nN(Prob<0.01) = {}'.format(len(predicted_prob), n_prob1, n_prob0), xy=(0.5, 0.8), xycoords='axes fraction', fontsize=16, horizontalalignment="center", verticalalignment='top')
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
        plt.vlines(my_thr, 0, 1.0, color='orange', label="Threshold")
    plot_precision_recall_vs_threshold(precision, recall, thresholds)

    _, id  = find_nearest(thresholds,my_thr)
    pt_pr = (id,thresholds[id],precision[id],recall[id])
    print("id,thr,pre,rec =",pt_pr)

    plt.savefig(f"{dirName}/TPRFcurve.pdf",bbox_inches='tight')
    plt.clf()

    plot_precision_recall_vs_threshold(precision, recall, thresholds)
    plt.xlim([0.8,1.0])
    plt.savefig(f"{dirName}/TPRFcurve_zoom.pdf",bbox_inches='tight')
    plt.clf()

    from sklearn.metrics import PrecisionRecallDisplay
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.plot(pt_pr[3], pt_pr[2], "o", label="Threshold")
    plt.ylim([0.0, 1.0])
    plt.savefig(f"{dirName}/PRcurve.pdf",bbox_inches='tight')
    plt.clf()

    from sklearn.metrics import roc_curve
    from sklearn.metrics import RocCurveDisplay
    fpr, tpr, thr = roc_curve(dataset_label, predicted_prob, pos_label=1)

    def plot_precision_recall_vs_threshold(fpr, tpr, thresholds):
        plt.plot(thresholds[1:], fpr[1:], "b--", label="False Positive Rate")
        plt.plot(thresholds[1:], tpr[1:], "r-", label="True Positive Rate")
        plt.xlabel("Threshold")
        plt.legend(loc="center right")
        plt.vlines(my_thr, 0, 1.0, color='orange', label="Threshold")
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
    
for thr in prob_thr:
    draw_test(test_loader, thr)
