#!/usr/bin/env bash

l=(5 7.5 10 12.5 15 17.5 20)
THR=0.95
MODEL="peak_finding_231118_104441.pth"

mkdir dataset
mkdir all_dataset_figs
# Dataset generation
for i in ${l[*]}; do
    echo Processing dataset ${i} ...
    mkdir dataset_figs
    time root -q -l -b data_gen_txt.C"(\"../rootfile/sample_$i.root\", \"dataset/dataset_for_test_$i.txt\")"
    mv -f dataset_figs all_dataset_figs/dataset_figs_${i}
done

# Test
mkdir test_results
mkdir test_results/test_figs
mkdir test_results/test_prob
mkdir test_results/test_log
for i in ${l[*]}; do
    echo Testing dataset ${i} ...
    mkdir test_results/test_figs/test_figs_$i
    (time python -u test.py -m=$MODEL -in=dataset/dataset_for_test_$i.txt -out=test_results/test_prob/prob_$i.root --cut=$THR --dir=test_results/test_figs/test_figs_$i) > test_results/test_log/test_$i.log 2>&1
done

# Analysis
mkdir ana_results
mkdir ana_results/ana_figs
mkdir ana_results/ana_pf_results
mkdir ana_results/ana_log
for i in ${l[*]}; do
    echo Processing counting dataset of rootfile ${i}...
    mkdir ana_results/ana_figs/ana_figs_$i
    (time python -u analysis.py --dataFile=rootfile/sample_$i.root --datasetFile=dataset/dataset_for_test_$i.txt --probFile=test_results/test_prob/prob_$i.root --output=ana_results/ana_pf_results/counting_test_$i.root --dir=ana_results/ana_figs/ana_figs_$i --cut=$THR) > ana_results/ana_log/ana_$i.log 2>&1
done
