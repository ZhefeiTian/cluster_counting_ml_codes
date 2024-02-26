#! /bin/bash

JOB_ID=$(date +"%y%m%d_%H%M%S")

mkdir train_results_$SLURM_JOB_ID

CUDA_VISIBLE_DEVICES=1 nohup time python -u train.py -in weighed_dataset.txt -n $SLURM_JOB_ID -e 50 -lr 0.0001 -bs 64 > train_results_$SLURM_JOB_ID/pf_$SLURM_JOB_ID.log 2>&1 & 
