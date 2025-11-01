# CNN–BiGRU Acoustic Model (ASR)

## Overview
This project implements an end-to-end Automatic Speech Recognition (ASR) system using a CNN–BiGRU backbone with CTC loss. It supports PCEN or log-Mel front-ends, SpecAugment, Time-Frequency Mixup, EMA, and curriculum learning.

## Structure
models/ : network architectures  
data/ : dataset + preprocessing  
utils/ : helpers, metrics, EMA  
optional_upgrades/ : beam search, LM, multi-GPU  
experiments/ : logs, checkpoints, plots  

## Run
python train_asr.py --data_root ./data/librispeech --epochs 20 --batch_size 16 --pcen 1 --use_se 1 --tf_mixup 1 --curriculum 1 --ema 1

## Novelty
Includes SE-CNN front-end, PCEN features, TF-Mixup, Curriculum learning, EMA stabilization, and beam search decoding.

