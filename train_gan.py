import torch
from tqdm import tqdm
import wandb
import os
import copy
from entmax import sparsemax
import torch.nn.functional as F


def one_hot(logits):
    # Step 1: Compute softmax probabilities
    softmax_probs = F.softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]

    # Step 2: Perform "hard" one-hot encoding in the forward pass
    hard_one_hot = torch.zeros_like(softmax_probs)

    hard_one_hot.scatter_(-1, softmax_probs.argmax(dim=-1, keepdim=True), 1.0)

    # Step 3: Combine hard one-hot with softmax for backpropagation
    one_hot_like = (hard_one_hot - softmax_probs).detach() + softmax_probs
    return one_hot_like

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

# 학습 함수 정의
def wassertein_train(discriminator, batch, device):
    input_ids = batch['input_probs'].float().to(device)
    labels = batch['label'].to(device)
    
    # output = discriminator(input_ids, attention_mask, target_ids_batch, input_masks_invert).squeeze(-1)
    output = discriminator(input_ids)
    
    if output.dim() > 1:  # Check if shape is [batch_size, 1]
        output = output.squeeze(-1)  # Remove the second dimension

    return output


def gan_trainer(tokenizer, num_epochs, generator, discriminator, g_optimizer, d_optimizer, dataloaders, loss_fn, device, g_lr_scheduler, d_lr_scheduler, save_name, save_path):
    wandb.init(project='EEG-Textv0.4', name=save_name)
    smoothing = 0.05
    g_best_loss = float('inf')
    d_best_loss = float('inf')
    g_save_name = ''
    d_save_name = ''
    
    for epoch in range(1, num_epochs + 1):

        for phase in ['train', 'dev']:
            if phase == 'train':
                generator.train()  # Set model to training mode
                discriminator.train()
            else:
                generator.eval()  # Set model to evaluate mode
                discriminator.eval()

            d_losses = []
            g_losses = []

            print('[{}]Epoch {}/{}'.format(phase, epoch, num_epochs))
            print('-' * 10)
            # Iterate over data.

            # input_masks는 target_ids를 기준으로 문자 있는 곳은 1 없는 곳은 0
            #target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
            # input_sample['target_ids'] = target_tokenized['input_ids'][0]
            # mask invert는 transformer encoder에 넣을 때 사용
            # input mask와 target mask는 Bart에 넣을 때 사용

            for (input_embeddings, seq_len, input_masks, input_mask_invert,
                    target_ids, target_mask, sentiment_labels) in tqdm(dataloaders[phase]):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
    
                """replace padding ids in target_ids with -100"""
                # target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
                target_ids_batch_converted = target_ids_batch.clone()
                target_ids_batch_converted[target_ids_batch_converted == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                # padding 위치 마스크 생성
                non_pad_mask = (target_ids_batch != tokenizer.pad_token_id).unsqueeze(-1)  # [batch_size, seq_len, 1]

                d_optimizer.zero_grad()
                g_optimizer.zero_grad()
            
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    batch_size = target_ids_batch.size(0)
                    # Vocabulary 크기 (토크나이저에서 가져옴)
                    vocab_size = tokenizer.vocab_size

                    r_labels = create_labels(batch_size, 1.0-smoothing, 1.0, device=device)
                    f_labels = create_labels(batch_size, 0.0, smoothing, device=device)
                    # fake 데이터 준비 (generator의 출력 사용)

                    fake_output = generator(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch_converted)
                    fake_output_logits = fake_output.logits.detach()  # [batch_size, seq_len, vocab_size]

                    fake_probs = one_hot(fake_output_logits)
                    # fake_probs = sparsemax(fake_output_logits, dim=-1)
                    
                    # # Gumbel-Softmax 적용하여 gen_probs 생성
                    # fake_probs = torch.nn.functional.gumbel_softmax(fake_output_logits, tau=1, hard=True)  # [batch_size, seq_len, vocab_size]
                    
                    # real 데이터 준비 (one-hot 벡터 사용)
                    real_probs = torch.nn.functional.one_hot(target_ids_batch, num_classes=vocab_size).float()
                    real_probs = real_probs * non_pad_mask  # padding 위치를 0으로 설정
                    
                    # pred_string = tokenizer.decode(fake_probs.argmax(dim=-1)[0], skip_special_tokens=True)
                    # real_string = tokenizer.decode(real_probs.argmax(dim=-1)[0], skip_special_tokens=True)
                    # print(f'[{phase}] pred_string: {pred_string}')
                    # print(f'[{phase}] real_string: {real_string}')
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

                    # Discriminator loss on real and fake data
                    _, real_loss = d_train(discriminator, real_encoding, loss_fn, device)
                    _, fake_loss = d_train(discriminator, fake_encoding, loss_fn, device)
                    d_loss = real_loss + fake_loss
                    # print(f'[{phase}] real_output ', real_output)
                    # print(f'[{phase}] fake_output ', fake_output)

                    # real_distance = wassertein_train(discriminator, real_encoding, device)
                    # fake_distance = wassertein_train(discriminator, fake_encoding, device)
                    
                    # d_loss = -(torch.mean(real_distance) - torch.mean(fake_distance))
                    
                    
                    if phase == 'train':
                        d_loss.backward()
                        d_optimizer.step()
            
                    # Generaotr 학습

                    discriminator.requires_grad_(False)

                    generated_output = generator(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch_converted)
                    output_logits = generated_output.logits
                    gen_probs = one_hot(output_logits)
                    # gen_probs = sparsemax(output_logits, dim=-1)

                    # gen_ids = torch.argmax(output_logits, dim=-1)
                    
                    # Generator loss (using discriminator's output on fake data)
                    
                    gen_labels = create_labels(batch_size, 1.0 - smoothing, 1.0, device=device)

                    gen_encoding = {
                            'input_probs': gen_probs,
                            'label': gen_labels,  # Labels are real (1) to fool the discriminator
                        }

                    _, g_loss = d_train(discriminator, gen_encoding, loss_fn, device)
                    # g_loss = -torch.mean(generated_output.loss + g_loss)
                    g_loss = g_loss
                    # print(f'[{phase}] output ', _)
                    # print(f'[{phase}] gen_labels ', gen_labels)
                    # gen_distance = wassertein_train(discriminator, gen_encoding, device)
                    # g_loss = - torch.mean(gen_distance)
                    
                    if phase == 'train':
                        g_loss.backward()
                        g_optimizer.step()

                    discriminator.requires_grad_(True)

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            g_loss_avg = sum(g_losses)/len(g_losses)
            d_loss_avg = sum(d_losses)/len(d_losses)

            if phase == 'dev':
                if g_loss_avg < g_best_loss:
                    g_best_loss = g_loss_avg
                    g_best_model = copy.deepcopy(generator.state_dict())
                    g_save_name = f'g_best_{epoch}_{save_name}.pt'
                    torch.save(g_best_model, os.path.join(save_path, g_save_name))
                    print(f'update best on dev checkpoint: {g_save_name} / g_loss: {g_loss_avg}')
                if d_loss_avg < d_best_loss:
                    d_best_loss = d_loss_avg
                    d_best_model = copy.deepcopy(discriminator.state_dict())
                    d_save_name = f'd_best_{epoch}_{save_name}.pt'
                    torch.save(d_best_model, os.path.join(save_path, d_save_name))
                    print(f'update best on dev checkpoint: {d_save_name} / d_loss: {d_loss_avg}')
                wandb.log({
                    f"{phase}_d_loss": d_loss_avg,
                    f"{phase}_g_loss": g_loss_avg
                }, step=epoch)
            elif phase == 'train':
                g_lr_scheduler.step()
                d_lr_scheduler.step()
                wandb.log({
                    f"{phase}_d_loss": d_loss_avg,
                    f"{phase}_g_loss": g_loss_avg,
                    "epoch": epoch,
                }, step=epoch)
            print(f"{phase} Epoch [{epoch}/{num_epochs}]  D Loss: {sum(d_losses)/len(d_losses)}  G Loss: {sum(g_losses)/len(g_losses)}")
    # generator.load_state_dict(torch.load(os.path.join(save_path, g_save_name)))
    # discriminator.load_state_dict(torch.load(os.path.join(save_path, d_save_name)))

    print('Train complete')
    wandb.finish()
    return generator, discriminator
