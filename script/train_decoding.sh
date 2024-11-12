nohup python3 train_decoding.py --model_name BrainTranslator \
    --task_name task1_task2_task3 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding > train_baseline_eeg.log &

nohup python3 main.py --model_name BrainTranslator \
    --task_name task1_task2_task3 \
    --num_epoch_step 20 \
    --train_input EEG \
    -lr 0.00002 \
    -b 32 \
    -s ./checkpoints/decoding \
    --save_name EEG-GAN \
    --cuda cuda:1> train_EEG-GAN_eeg.log &