import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
from transformers import BartTokenizer, BartForConditionalGeneration, \
    BertTokenizer, BertConfig

from data import ZuCo_dataset
from model_decoding import BrainTranslator, BartDiscriminator, Critic, RNNDiscriminator
from config import get_config
from train_gan import gan_trainer
from eval_decoding import eval_model

if __name__ == '__main__':

    # 1. Gen pre-train
    args = get_config('main')
    pre_trained_g = args['pre_trained_g']
    pre_trained_d = args['pre_trained_d']

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs = args['num_epoch_step'] # default: 30
    lr = args['learning_rate_step'] # default: 0.0000005

    batch_size = args['batch_size'] # default: 32

    model_name = args['model_name'] # default: BrainTranslator
    save_name = args['save_name']
    train_input = args['train_input']
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name'] # default: task1_task2_taskNRv2
    device_ids = [0] # device setting

    save_path = args['save_path'] # default: checkpoints/decoding
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(f'[INFO]using model: {model_name}')

    g_pretrained_save_name = f'pretrained_g_{save_name}'
    d_pretrained_save_name = f'pretrained_d_{save_name}'

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{g_pretrained_save_name}.pt')
    output_checkpoint_name_best_d = os.path.join(save_path_best, f'{d_pretrained_save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{g_pretrained_save_name}.pt')
    output_checkpoint_name_last_d = os.path.join(save_path_last, f'{d_pretrained_save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')



    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        # dev = "cuda:3" 
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/mnt/data/members/speech/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/mnt/data/members/speech/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/mnt/data/members/speech/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/mnt/data/members/speech/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True

    # ---------------------------------------------------------------------------------------
    # Data Loader

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dev dataloader
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader,'test':test_dataloader}


    # --------------------------------------------------------------------------------
    # Generator Define
    ''' set up model '''
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        generator = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    generator.to(device)
    generator = torch.nn.DataParallel(generator, device_ids=device_ids)
    ''' set up optimizer and scheduler'''
    g_optimizer = AdamW(generator.parameters(), lr=lr, weight_decay=0.01)

    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.1)

    ''' set up loss function '''
    # loss_fn = nn.BCEWithLogitsLoss()
    # BCEWithLogitsLoss는 내부에서 sigmoid를 적용한 후 BCELoss를 적용하는데
    # discriminator의 출력층에 sigmoid를 적용하고 있기 때문에
    # BCELoss를 사용해야 한다.
    loss_fn = nn.BCELoss()

    # --------------------------------------------------------------------------------
    # Discriminator Define
    vocab_size = tokenizer.vocab_size
    
    discriminator = BartDiscriminator()
    # discriminator = Critic()
    discriminator = discriminator.to(device)
    discriminator = torch.nn.DataParallel(discriminator, device_ids=device_ids)

    d_optimizer = Adam(discriminator.parameters(), lr=lr/10)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=15, gamma=0.1)

    max_len=512

    # --------------------------------------------------------------------------------
    # Train
    generator, discriminator = gan_trainer(tokenizer,num_epochs, generator, discriminator, g_optimizer, d_optimizer, dataloaders, loss_fn, device, g_lr_scheduler, d_lr_scheduler, save_name, save_path_best)

    g_save_name = f'generator_{save_name}.pt'
    d_save_name = f'discriminator_{save_name}.pt'

    # eval
    eval_model(dataloaders, device, tokenizer, generator, output_all_results_path = f'./results/{g_save_name}_results.txt' , score_results=f'./score_results/{g_save_name}_results.txt')
    
    torch.save(generator.state_dict(), os.path.join(save_path_last, f'{g_save_name}.pt'))
    torch.save(discriminator.state_dict(), os.path.join(save_path_last, f'{d_save_name}.pt'))