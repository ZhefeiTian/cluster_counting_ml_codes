#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import ROOT
from ROOT import TFile, TTree
from array import array
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', type=str, default='signal_10.root')
parser.add_argument('--datasetFile', type=str, default='dataset_for_test.txt')
parser.add_argument('--probFile', type=str, default='prob.root')
parser.add_argument('--output', type=str, default='../3_cluster_train/counting_test.root')
parser.add_argument('--nsize', type=int, default=-1)
parser.add_argument('--isData', type=bool, default=False)
parser.add_argument('--useD2', type=bool, default=False)
parser.add_argument('--cut', type=float, default=0.999)#0.99999
parser.add_argument('--dir', '-d', type=str, default='ana')

#args = parser.parse_args([])
args = parser.parse_args()

filename_data = args.dataFile
filename_dataset = args.datasetFile
filename_prob = args.probFile
filename_output = args.output
nsize = args.nsize
isData = args.isData
useD2 = args.useD2
cut = args.cut
dirName = args.dir

# In[2]:


global fig_id # ADDED
fig_id = 0
def plot_waveform(wf, amp, time, tag, truth, truth_tag):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel('Index')
    ax.set_ylabel('Amplitude')

    wf_x = list(range(len(wf)))
    ax.plot(wf_x, wf)
# 0=bkg, 1=pri, 2=sec
    if not tag is None:
        amp_pri = [a for i, a in enumerate(amp) if tag[i] == 1] 
        time_pri = [t for i, t in enumerate(time) if tag[i] == 1]
        ax.plot(time_pri, amp_pri, 'o')

        amp_sec = [a for i, a in enumerate(amp) if tag[i] == 2]
        time_sec = [t for i, t in enumerate(time) if tag[i] == 2]
        ax.plot(time_sec, amp_sec, 'o')

        amp_wrong = [a for i, a in enumerate(amp) if tag[i] == 0]
        time_wrong = [t for i, t in enumerate(time) if tag[i] == 0]
        ax.plot(time_wrong, amp_wrong, 'o')

        truth_time_pri = [t for i, t in enumerate(truth) if truth_tag[i] == 0]
        truth_amp_pri = [wf[int(t)] for t in truth_time_pri]
        ax.vlines(truth_time_pri, 0, truth_amp_pri, color='orange')

        truth_time_sec = [t for i, t in enumerate(truth) if truth_tag[i] == 1]
        truth_amp_sec = [wf[int(t)] for t in truth_time_sec]
        ax.vlines(truth_time_sec, 0, truth_amp_sec, color='green')

    if tag is None:
        ax.plot(time, amp, 'o')

    global fig_id # ADDED
    plt.savefig(f'{dirName}/counting_%d.pdf' % (fig_id))
    plt.clf()
    fig_id = fig_id + 1

def sort_map(this_map):
    sorted_item = sorted(this_map.items())
    return {k:v for k,v in sorted_item}

def match(truth, truth_tag, det):
    det2truth = []
    for i, t_det in enumerate(det):
        this_map = {}
        for j, t_truth in enumerate(truth):
            this_map[abs(t_det - t_truth)] = j
        sorted_map = sort_map(this_map)
        det2truth.append(sorted_map)

    truth2det = []
    for i, t_truth in enumerate(truth):
        this_map = {}
        for j, t_det in enumerate(det):
            this_map[abs(t_det - t_truth)] = j
        sorted_map = sort_map(this_map)
        truth2det.append(sorted_map)


    id = [0 for x in det]

    for i, d2t_map in enumerate(det2truth):
        truth_idx = list(d2t_map.values())[0]

        t2d_map = truth2det[truth_idx]
        det_idx = list(t2d_map.values())[0]

        if i == det_idx:
            id[i] = truth_tag[truth_idx]

    return id

# data file
file_data = TFile(filename_data)
tree = file_data.Get('sim')

sampling_rate = array('d', [-1])
wf = ROOT.std.vector['double'](0)
truth_time = ROOT.std.vector['double'](0)
truth_tag = ROOT.std.vector['int'](0)

tree.SetBranchAddress('sampling_rate', sampling_rate)
tree.SetBranchAddress('wf_i', wf)
if not isData:
    tree.SetBranchAddress('time', truth_time)
    tree.SetBranchAddress('tag', truth_tag)

# dataset file
nleft = 5
nright = 9
ndim = 1

dataset = pd.read_csv(filename_dataset, skipinitialspace=True)

evtno_list = dataset['EvtNo'].values
peak_time_list = dataset['Time']


# prob file
file_prob = TFile(filename_prob)
tree_prob = file_prob.Get('tmva')

prob_ml = array('d', [-1.])
prob_d2 = array('d', [-1.])

tree_prob.SetBranchAddress('prob_rnn', prob_ml)
if useD2:
    tree_prob.SetBranchAddress('prob_d2', prob_d2)

# output file
file_out = TFile(filename_output, 'recreate')
tree_out = TTree('sim', 'sim')

ncount = array('i', [-1])
ncount_pri = array('i', [-1])
ncls = array('i', [-1])
sampling_rate_out = array('d', [-1.])
count_x = ROOT.std.vector['double'](0)
id = ROOT.std.vector['int'](0)
wf_out = ROOT.std.vector['double'](0)

tree_out.Branch('sampling_rate', sampling_rate_out, 'sampling_rate/D')
tree_out.Branch('wf_i', wf_out)
tree_out.Branch('ncount', ncount, 'ncount/I')
tree_out.Branch('count_x', count_x)
if not isData:
    tree_out.Branch('id', id)
    tree_out.Branch('ncount_pri', ncount_pri, 'ncount_pri/I')
    tree_out.Branch('ncls', ncls, 'ncls/I')

n = tree_prob.GetEntries() if nsize < 0 else nsize

detected_time = {}
for i in range(n):
    if i % 1000000 == 0:
        print('Processing event %d ...' % i)
    
    tree_prob.GetEntry(i)

    prob = prob_d2[0] if useD2 else prob_ml[0]
    evtno = evtno_list[i]
    peak_time = peak_time_list[i]
    if cut == 1:
        if (prob == 1.): # modified
            if not evtno in detected_time:
                detected_time[evtno] = []
            detected_time[evtno].append(peak_time)
    else:
        if (prob > cut): # modified
            if not evtno in detected_time:
                detected_time[evtno] = []
            detected_time[evtno].append(peak_time)

#print(detected_time)

fig_id = 0
for evtno in detected_time:
    tree.GetEntry(evtno)

    sampling_rate_out[0] = sampling_rate[0]
    vtime = detected_time[evtno]
    vamp = []
    ncount[0] = len(vtime)
    count_x.clear()
    wf_out.clear()
    id.clear()

    for i in range(wf.size()):
        if isData:
            wf[i] = wf[i] * -1.
        wf_out.push_back(wf[i])

    for t in vtime:
        vamp.append(wf[int(t)])
        count_x.push_back(float(t/sampling_rate))

    vtime_truth = []
    vtag_truth = []
    if not isData:
        ncls[0] = 0
        for itruth in range(truth_tag.size()):
            if truth_tag[itruth] > 0:
                vtime_truth.append(truth_time[itruth])
                vtag_truth.append(truth_tag[itruth])
            if truth_tag[itruth] == 1:
                ncls[0] = ncls[0] + 1

        id_list = match(vtime_truth, vtag_truth, vtime)
        for _id in id_list:
            id.push_back(_id)
            
        ncount_pri[0] = 0
        for idx in range(id.size()):
            if id[idx] == 1:
                ncount_pri[0] = ncount_pri[0] + 1

    if evtno < 10:
        wf_list = []
        for iwf in range(wf.size()):
            wf_list.append(wf[iwf])
        
        if not isData:
            plot_waveform(wf_list, vamp, vtime, id_list, vtime_truth, vtag_truth)
        else:
            plot_waveform(wf_list, vamp, vtime, None, None, None)


    tree_out.Fill()

file_out.WriteTObject(tree_out)
file_out.Close()

precision = []
for evtno in detected_time:
    tree.GetEntry(evtno)

    sampling_rate_out[0] = sampling_rate[0]
    vtime = detected_time[evtno]
    vamp = []
    ncount[0] = len(vtime)

    vtime_truth = []
    vtag_truth = []
    if not isData:
        ncls[0] = 0
        
        for itruth in range(truth_tag.size()):
            if truth_tag[itruth] > 0:
                vtime_truth.append(truth_time[itruth])
                vtag_truth.append(truth_tag[itruth])

        id_list = match(vtime_truth, vtag_truth, vtime)
    precision.append([id_list.count(1),id_list.count(2),id_list.count(0),vtag_truth.count(1),vtag_truth.count(2)])
    
precision_rotation=np.array(precision).T

FindingPri = precision_rotation[0]
FindingSec = precision_rotation[1]
FindingWrong = precision_rotation[2]
TruthPri = precision_rotation[3]
TruthSec = precision_rotation[4]

FindingAll = FindingPri + FindingSec + FindingWrong
TruthAll = TruthPri + TruthSec

print(sum(FindingAll),sum(FindingPri),sum(FindingSec),sum(FindingWrong))
print(FindingPri+FindingSec)
print(precision_rotation)

import scipy.stats

plt.clf()
plt.figure(figsize=(6.4, 4.8))

countging_efficiency = FindingAll/TruthAll
bins = np.arange(0.225, 1.375, 0.05)
n, _, _ =plt.hist(countging_efficiency, bins, histtype='step', label="Counting efficiency")
area=np.sum(0.05*n)

mu, sigma = scipy.stats.norm.fit(countging_efficiency) # mu is set to mean
print(mu, sigma)
curve_bins = bins
best_fit_line = scipy.stats.norm.pdf(curve_bins, mu, sigma)
plt.plot(curve_bins, area*best_fit_line, 'r--', label="Fitting")

plt.plot([], [], ' ', label="μ = "+str(round(mu,5)))
plt.plot([], [], ' ', label="σ = "+str(round(sigma,5)))
plt.legend()
plt.xlabel("Counting Efficiency")

plt.savefig(f"{dirName}/CountingEfficiency.pdf")
plt.clf()

bins = np.arange(0, 67, 1)
n, _, _ =plt.hist(TruthAll, bins, histtype='step', color="r",  label="# of Peaks (Truth)")
plt.xlabel("# of Total Peaks (Truth)")


plt.savefig(f"{dirName}/TruthPeaks.pdf")
plt.clf()

bins = np.arange(0, 100, 1)
n, _, _ =plt.hist(TruthPri, bins, histtype='step', color="r", label="Truth of Peaks")

plt.xlabel("# of Primary Ionization (Truth)")
#plt.show()
plt.savefig(f"{dirName}/TruthPrimaryIonization.pdf")
plt.clf()

bins = np.arange(0, 100, 1)
n, _, _ =plt.hist(TruthSec, bins, histtype='step', color="r", label="Truth of Peaks")

plt.xlabel("# of Second Ionization (Truth)")
#plt.show()
plt.savefig(f"{dirName}/TruthSecondIonization.pdf")
plt.clf()

bins = np.arange(0, 100, 1)
n, _, _ =plt.hist(FindingAll, bins, histtype='step', label="# of Total Peaks (ML)")
plt.xlabel("# of Total Peaks (ML)")
plt.savefig(f"{dirName}/PeakFindingResults.pdf")
plt.clf()

bins = np.arange(0, 100, 1)
plt.hist(TruthAll, bins, histtype='step',  alpha=0.5, color="r", label="Truth")
plt.hist(FindingAll, bins, histtype='step',  alpha=0.5, color="g", label="LSTM model")
plt.legend()
plt.xlabel("# of Total Peaks in Event")
plt.savefig(f"{dirName}/ComprisionOfPeaks.pdf")
plt.clf()

bins = np.arange(0, 100, 1)
plt.hist(TruthPri, bins, histtype='step',  alpha=0.5, color="r", label="Truth")
plt.hist(FindingPri, bins, histtype='step',  alpha=0.5, color="g", label="LSTM model")
plt.legend()
plt.xlabel("# of Primary Ionization in Event")
plt.savefig(f"{dirName}/ComprisionOfPri.pdf")
plt.clf()

bins = np.arange(0, 100, 2)
plt.hist(0.7*TruthPri, bins, histtype='step',  alpha=0.5, color="r", label="0.7×Truth")
plt.hist(FindingPri, bins, histtype='step',  alpha=0.5, color="g", label="LSTM model")
plt.legend()
plt.xlabel("# of Primary Ionization in Event")
plt.savefig(f"{dirName}/ComprisionOfPri_2.pdf")
plt.clf()

wrong_rate = FindingWrong/FindingAll

wrong_rate_bins = np.arange(0.0, 0.25, 0.0125)
#[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
plt.hist(wrong_rate, wrong_rate_bins, histtype='step', color="r")
plt.xlabel("False Discovery Rate")
#plt.legend()
plt.savefig(f"{dirName}/FalseDiscoveryRate.pdf")
plt.clf()

from matplotlib.ticker import MaxNLocator
def PlotNum2FDR(arr):
    FindingNum = np.unique(arr)
    height_of_Finding = np.array([np.sum(arr == i) for i in FindingNum])

    print(FindingNum)
    fdr_for_num_aver = []
    for i in FindingNum:
        fdr_for_num_array = wrong_rate[arr==i]
        fdr_for_num_aver.append(np.mean(fdr_for_num_array))
    factor = np.max(fdr_for_num_aver)/np.max(height_of_Finding)
    plt.plot(FindingNum,fdr_for_num_aver,marker="o",label="False Discovery Rate")
    plt.plot(FindingNum,factor*height_of_Finding,marker="o",label="Number of Events (scaling)")

PlotNum2FDR(FindingPri)
plt.xlabel("Number of Primary Ionization (Found)")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.savefig(f"{dirName}/FDR_to_FindingPri.pdf")
plt.clf()

PlotNum2FDR(TruthPri)
plt.xlabel("Number of Primary Ionization (Truth)")
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()
plt.savefig(f"{dirName}/FDR_to_TruthPri.pdf")
plt.clf()
