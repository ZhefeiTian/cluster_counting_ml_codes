#!/usr/bin/env bash

l=(5 7.5 10 12.5 15 17.5 20)
THR=0.26
MODEL="cls_model_full_240122_095619.pth"
ln -sf ../test_PeakFinding/ana_results/ana_pf_results pf_results

mkdir dataset
# Dataset generation
for i in ${l[*]}; do
    echo Processing dataset ${i} ...
    time root -q -l -b data_gen_total_test.C"(\"./pf_results/counting_test_$i.root\", \"./dataset/peaks_$k.txt\", \"./dataset/times_$k.txt\")"
done

# Test
mkdir cls_results
mkdir cls_results/results/result_$i
mkdir cls_results/log
for i in ${l[*]}; do
    echo Testing dataset ${i} ...
    mkdir cls_results/results/${k}
    (time python -u test.py -dst=dataset/times_$i.txt -dsp=dataset/peaks_$i.txt -sample=../rootfile/sample_$i.root -i=pf_results/counting_test_$i.root -t=$i -d=cls_results/results/result_$i -m=$MODEL) > cls_results/log/cls_test_$i.log 2>&1
done
