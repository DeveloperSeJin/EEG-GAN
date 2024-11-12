import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import copy
from tqdm import tqdm
from data import ZuCo_dataset
from model_decoding import BrainTranslator, DDataset, BART_BiLSTM
from config import get_config
from train_gan import gan_trainer
import wandb
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, \
    BertTokenizer, BertConfig, RobertaTokenizer


# 학습 함수 정의
def d_train(discriminator, batch, device):
    input_ids = batch['input_ids'].float().to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = torch.tensor([batch['label']], dtype=torch.float).squeeze(0).to(device)
    target_ids_batch = batch['target_ids_batch'].to(device)
    input_masks_invert = batch['input_masks_invert'].to(device)
    
    output = discriminator(input_ids, attention_mask, target_ids_batch, input_masks_invert).squeeze(-1)

    return output


def gan_trainer(g_tokenizer, d_tokenizer, num_epochs, generator, discriminator, g_optimizer, d_optimizer, dataloaders, d_loss_fn, g_loss_fn, device, g_lr_scheduler, d_lr_scheduler, save_name, save_path):
    wandb.init(project='EEG-Text', name=save_name)

    g_best_loss = float('inf')
    d_best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)


        for phase in ['train', 'dev']:
            if phase == 'train':
                generator.train()  # Set model to training mode
                discriminator.train()
            else:
                generator.eval()  # Set model to evaluate mode
                discriminator.eval()

            d_losses = []
            g_losses = []

            # Iterate over data.
            for (input_embeddings, seq_len, input_masks, input_mask_invert,
                    target_ids, target_mask, sentiment_labels) in tqdm(dataloaders[phase]):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
    
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == g_tokenizer.pad_token_id] = -100
                # zero the parameter gradients

                d_optimizer.zero_grad()
                g_optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    
                    fake_output = generator(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                    output_logits = fake_output.logits
                    fake_ids = torch.argmax(output_logits, dim=-1)
                    # print('fake ids: ', fake_ids.shape)
                    # print('target ids: ', target_ids_batch.shape)



                    r_labels = np.ones(len(target_ids_batch))
                    f_labels = np.zeros(len(fake_ids))

                    # Prepare real data for discriminator
                    real_encoding = {
                        'input_ids': target_ids_batch,
                        'attention_mask': input_masks_batch,
                        'label': r_labels,
                        'target_ids_batch': target_ids_batch,
                        'input_masks_invert': input_mask_invert_batch
                    }

                    # Prepare fake data for discriminator
                    fake_encoding = {
                        'input_ids': fake_ids.detach(),
                        'attention_mask': input_masks_batch,
                        'label': f_labels,
                        'target_ids_batch': target_ids_batch,
                        'input_masks_invert': input_mask_invert_batch
                    }

                    # Discriminator loss on real and fake data
                    real_output = d_train(discriminator, real_encoding, device)
                    fake_output = d_train(discriminator, fake_encoding, device)
                    d_loss = -torch.mean(real_output) + torch.mean(fake_output)

                    if phase == 'train':
                        with torch.autograd.detect_anomaly():
                            d_loss.backward()
                            d_optimizer.step()

                    d_losses.append(d_loss.item())

                    if epoch % 5 == 0:
                        generated_output = generator(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                        output_logits = generated_output.logits
                        gen_ids = torch.argmax(output_logits, dim=-1)
                        
                        # Generator loss (using discriminator's output on fake data)
                        gen_encoding = {
                                'input_ids': gen_ids,
                                'attention_mask': input_masks_batch,
                                'label': r_labels,  # Labels are real (1) to fool the discriminator
                                'target_ids_batch': target_ids_batch,
                                'input_masks_invert': input_mask_invert_batch
                            }

                        g_output = d_train(discriminator, gen_encoding, device)
                        g_loss = -torch.mean(g_output)

                        if phase == 'train':
                            with torch.autograd.detect_anomaly():
                                g_loss.backward()
                                g_optimizer.step()
                        g_losses.append(g_loss.item())

            if epoch % 5 == 0:
                if phase == 'dev':
                    wandb.log({
                        f"{phase}_d_loss": sum(d_losses)/len(d_losses),
                        f"{phase}_g_loss": sum(g_losses)/len(g_losses),
                        "epoch": epoch
                    })
                elif phase == 'train':
                    g_lr_scheduler.step()
                    d_lr_scheduler.step()
                    wandb.log({
                        f"{phase}_d_loss": sum(d_losses)/len(d_losses),
                        f"{phase}_g_loss": sum(g_losses)/len(g_losses),
                        "epoch": epoch
                    })
                print(f"{phase} Epoch [{epoch}/{num_epochs}]  D Loss: {sum(d_losses)/len(d_losses)}  G Loss: {sum(g_losses)/len(g_losses)}")

        if epoch % 5 == 0:
            if phase == 'dev' and g_loss < g_best_loss:
                g_best_loss = g_loss
                g_best_model = copy.deepcopy(generator.state_dict())
                torch.save(generator.state_dict(), os.path.join(save_path, f'g_{epoch}_{save_name}.pt'))
                print(f'update best on dev checkpoint: g_{epoch}_{save_name}.pt')
                wandb.run.summary["g_best_loss"] = g_best_loss
            if phase == 'dev' and d_loss < d_best_loss:
                d_best_loss = d_loss
                d_best_model = copy.deepcopy(discriminator.state_dict())
                torch.save(discriminator.state_dict(), os.path.join(save_path, f'd_{epoch}_{save_name}.pt'))
                print(f'update best on dev checkpoint: d_{epoch}_{save_name}.pt')
                wandb.run.summary["d_best_loss"] = d_best_loss


    
    wandb.finish()
    generator.load_state_dict(g_best_model)
    discriminator.load_state_dict(d_best_model)
    return generator, discriminator

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
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
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
        g_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        g_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True

    # ---------------------------------------------------------------------------------------
    # Data Loader

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', g_tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', g_tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}


    # --------------------------------------------------------------------------------
    # Generator Define
    ''' set up model '''
    if model_name == 'BrainTranslator':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        generator = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    generator.to(device)

    ''' set up optimizer and scheduler'''
    g_optimizer = optim.SGD(generator.parameters(), lr=lr, momentum=0.9)

    g_lr_scheduler = lr_scheduler.StepLR(g_optimizer, step_size=30, gamma=0.1)

    ''' set up loss function '''
    g_loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------------
    # Discriminator Define
    d_tokenizer = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    d_loss_fn = BCEWithLogitsLoss()
    discriminator = BART_BiLSTM()
    discriminator = discriminator.to(device)
    d_optimizer = AdamW(discriminator.parameters(), lr=0.000002)
    d_lr_scheduler = lr_scheduler.StepLR(d_optimizer, step_size=15, gamma=0.1)

    max_len=512

    # --------------------------------------------------------------------------------
    # Train
    generator, discriminator = gan_trainer(g_tokenizer, d_tokenizer,num_epochs, generator, discriminator, g_optimizer, d_optimizer, dataloaders, d_loss_fn, g_loss_fn, device, g_lr_scheduler, d_lr_scheduler, save_name, save_path_best)

    g_save_name = f'generator_{save_name}.pt'
    d_save_name = f'discriminator_{save_name}.pt'

    torch.save(generator.state_dict(), os.path.join(save_path_last, f'{g_save_name}.pt'))
    torch.save(discriminator.state_dict(), os.path.join(save_path_last, f'{d_save_name}.pt'))