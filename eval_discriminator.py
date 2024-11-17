from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from model_decoding import BrainTranslator, Discriminator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
from glob import glob
from transformers import BartTokenizer, BartForConditionalGeneration
from data import ZuCo_dataset
from model_decoding import BrainTranslator
from config import get_config
from tqdm import tqdm

def create_labels(n: int, r1: float, r2: float, device: torch.device = None):
    """
    Create smoothed labels
    """
    return torch.empty(n, 1, requires_grad=False, device=device).uniform_(r1, r2).squeeze(1)

# 학습 함수 정의
def d_train(discriminator, batch, loss_fn, device):
    input_ids = batch['input_probs'].float().to(device)
    labels = batch['label'].to(device)
    
    # output = discriminator(input_ids, attention_mask, target_ids_batch, input_masks_invert).squeeze(-1)
    output = discriminator(input_ids)
    
    if output.dim() > 1:  # Check if shape is [batch_size, 1]
        output = output.squeeze(-1)  # Remove the second dimension
        
    loss = loss_fn(output, labels)
    return output.detach(), loss

# 평가 함수 정의
def evaluate(generator, discriminator, dataloaders, loss_fn, device):
    discriminator.eval()
    model.eval()  # 평가 모드로 전환
    criterion = nn.BCELoss()  # 이진 크로스 엔트로피 손실
    all_preds = []
    all_labels = []
    smoothing = 0.05
    total_loss = 0.0
    # with open('./results/discriminator_results.txt','w') as f:
    for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders['test']):

        input_embeddings_batch = input_embeddings.to(device).float() # B, 56, 840
        input_masks_batch = input_masks.to(device).bool() # B, 56
        target_ids_batch = target_ids.to(device) # B, 56
        input_mask_invert_batch = input_mask_invert.to(device).bool() # B, 56
        
        target_ids_batch_converted = target_ids_batch.clone()
        target_ids_batch_converted[target_ids_batch_converted == tokenizer.pad_token_id] = -100

        # zero the parameter gradients
        # padding 위치 마스크 생성
        non_pad_mask = (target_ids_batch != tokenizer.pad_token_id).unsqueeze(-1)  # [batch_size, seq_len, 1]

        batch_size = target_ids_batch.size(0)
        # Vocabulary 크기 (토크나이저에서 가져옴)
        vocab_size = tokenizer.vocab_size

        r_labels = create_labels(batch_size, 1.0-smoothing, 1.0, device=device)
        f_labels = create_labels(batch_size, 0.0, smoothing, device=device)
        # fake 데이터 준비 (generator의 출력 사용)

        fake_output = generator(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch_converted)
        fake_output_logits = fake_output.logits.detach()  # [batch_size, seq_len, vocab_size]

        # Gumbel-Softmax 적용하여 gen_probs 생성
        fake_probs = torch.nn.functional.gumbel_softmax(fake_output_logits, tau=1, hard=True)  # [batch_size, seq_len, vocab_size]

        # real 데이터 준비 (one-hot 벡터 사용)
        real_probs = torch.nn.functional.one_hot(target_ids_batch, num_classes=vocab_size).float()
        real_probs = real_probs * non_pad_mask  # padding 위치를 0으로 설정
        # Prepare real data for discriminator
        real_encoding = {
            'input_probs': real_probs,
            'label': r_labels,
        }

        # Prepare fake data for discriminator
        fake_encoding = {
            'input_probs': fake_probs,
            'label': f_labels,
        }

        real_output, real_loss = d_train(discriminator, real_encoding, loss_fn, device)
        fake_output, fake_loss = d_train(discriminator, fake_encoding, loss_fn, device)
        
        outputs = torch.cat([real_output, fake_output], dim=0)
        labels = torch.cat([r_labels, f_labels], dim=0) 

        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss = real_loss + fake_loss
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloaders['test'])

    # 이진화된 실제 라벨 생성
    binary_labels = [1 if l >= 0.5 else 0 for l in all_labels]
    # 정확도 계산
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    print('binary_labels: ',binary_labels)
    print('binary_preds: ', binary_preds)
    accuracy = accuracy_score(binary_labels, binary_preds)
    # 정밀도, 재현율, F1 점수 계산
    precision, recall, f1, _ = precision_recall_fscore_support(binary_labels, binary_preds, average='binary')
    # ROC-AUC 계산
    roc_auc = roc_auc_score(binary_labels, binary_preds)
    
    print(f'평균 손실: {avg_loss:.4f}')
    print(f'정확도: {accuracy:.4f}')
    print(f'정밀도: {precision:.4f}')
    print(f'재현율: {recall:.4f}')
    print(f'F1 점수: {f1:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# 메인 실행 부분
if __name__ == '__main__':
    batch_size = 1
    ''' get args'''
    args = get_config('eval_decoding')
    test_input = args['test_input']
    print("test_input is:", test_input)
    train_input = args['train_input']
    print("train_input is:", train_input)
    ''' load training config'''
    training_config = json.load(open(args['config_path']))


    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']
    model_name = training_config['model_name']
    

    if test_input == 'EEG' and train_input=='EEG':
        print("EEG and EEG")
        output_all_results_path = f'./results/{task_name}-{model_name}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}.txt'
    else:
        output_all_results_path = f'./results/{task_name}-{model_name}-{train_input}_{test_input}-all_decoding_results.txt'
        score_results = f'./score_results/{task_name}-{model_name}-{train_input}_{test_input}.txt'


    ''' set random seeds '''
    seed_val = 20 #500
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = args['cuda'] 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    # task_name = 'task1_task2_task3'

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
    
    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    
    if model_name == 'BrainTranslator':
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
   

    state_dict = torch.load(checkpoint_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    '''
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path))
    '''

    # model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    vocab_size = tokenizer.vocab_size
    discriminator = Discriminator(vocab_size=vocab_size, embedding_dim=512, hidden_dim=128)

    discriminator_state_dict = torch.load('./checkpoints/decoding/best/d_20_del.pt')
    discriminator.load_state_dict(discriminator_state_dict)
    discriminator.to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    # 평가 수행
    metrics = evaluate(model, discriminator, dataloaders, loss_fn, device)
