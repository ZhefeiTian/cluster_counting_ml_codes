#! /bin/bash

JOB_ID=$(date +"%y%m%d_%H%M%S")

mkdir cls_train_results_$JOB_ID
mkdir cls_train_results_$JOB_ID/ckpt
echo "Directory created..."
echo ""

CUDA_VISIBLE_DEVICES=1 nohup time python -u train.py -dsp dataset/ALL_peaks.txt -dst dataset/ALL_times.txt -d cls_train_results_$SLURM_JOB_ID -lr 0.001 -bs 128 -e 100 > cls_train_results_$SLURM_JOB_ID/cls_$SLURM_JOB_ID.log 2>&1 &
