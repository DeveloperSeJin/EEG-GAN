#!/bin/bash

# Task 배열 정의
tasks=("task1_task2_task3_taskNRv2" "task1_task2_taskNRv2" "task2_taskNRv2")

# 각 Task에 대해 반복
for t in "${tasks[@]}"; do
	nohup python3 main.py --model_name BrainTranslator \
	--task_name "$t" \
       	--num_epoch_step 20 \
	--train_input EEG \
	-lr 0.00002 \
	-b 32 \
	-s ./checkpoints/decoding \
	--save_name EEG-GAN_"$t" \
	--cuda cuda:1> EEG-GAN_"$t".log &
done
