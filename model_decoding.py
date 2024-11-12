import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, RobertaModel, RobertaTokenizer
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration

class BART_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=256, dropout_prob=0.3, decoder_embedding_size=1024):
        super(BART_BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # BART 모델 로드
        self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        # Transformer Encoder Layer 설정
        self.additional_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bart.config.d_model,  # BART의 embedding dimension 사용
            nhead=8,
            dim_feedforward=decoder_embedding_size,
            batch_first=True
        )
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=self.bart.config.d_model,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Dropout 레이어
        self.dropout = nn.Dropout(dropout_prob)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, gen_probs, attention_mask, target_ids_batch_converted, input_masks_invert):
        # BART의 embedding matrix 가져오기
        embedding_matrix = self.bart.model.shared.weight  # [vocab_size, embed_dim]

        # gen_probs와 embedding_matrix의 내적 계산
        embeddings = torch.matmul(gen_probs, embedding_matrix)  # [batch_size, seq_len, embed_dim]

        # Transformer Encoder
        encoded_embedding = self.additional_encoder(embeddings, src_key_padding_mask=input_masks_invert)

        # BART 모델
        out = self.bart(
            inputs_embeds=encoded_embedding,
            attention_mask=attention_mask,
            return_dict=True,
            labels=target_ids_batch_converted,
            output_hidden_states=True
        )

        # 마지막 4개의 encoder hidden states 합산
        encoder_hidden_states = out.encoder_hidden_states
        last_4_layers = encoder_hidden_states[-4:]
        output = sum(last_4_layers)

        # LSTM 입력
        lstm_out, _ = self.lstm(output)

        # BiLSTM의 출력 결합
        lstm_out_forward = lstm_out[:, -1, :self.hidden_dim]
        lstm_out_backward = lstm_out[:, 0, self.hidden_dim:]
        lstm_out_combined = torch.cat((lstm_out_forward, lstm_out_backward), dim=1)

        # Dropout 적용
        lstm_out_combined = self.dropout(lstm_out_combined)

        # Fully Connected Layer로 예측값 생성
        logits = self.fc(lstm_out_combined)
        output = torch.sigmoid(logits)

        return output


class Discriminator(nn.Module):
    def __init__(self, vocab_size, num_filters=128, kernel_sizes=[3, 4, 5], dropout_prob=0.3):
        super(Discriminator, self).__init__()
        
        # Convolutional layers with different kernel sizes to capture varying n-gram features
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, vocab_size)) for k in kernel_sizes
        ])
        
        # Fully connected layer
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x: [batch_size, seq_len, vocab_size] 형태의 one-hot 인코딩된 입력
        """
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, vocab_size]
        
        # Apply convolution and ReLU, followed by max pooling over sequence length
        conv_results = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs  # [batch_size, num_filters, seq_len - k + 1]
        ]
        pooled_results = [
            F.max_pool1d(conv_result, conv_result.size(2)).squeeze(2) for conv_result in conv_results
        ]  # [batch_size, num_filters] for each kernel size
        
        # Concatenate pooled features from different kernel sizes
        concat = torch.cat(pooled_results, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]
        
        # Apply dropout and fully connected layer to get logit
        out = self.dropout(concat)
        logit = self.fc(out).squeeze(1)  # [batch_size]
        
        # Apply sigmoid to get probability (0-1)
        output = torch.sigmoid(logit)
        
        return output


    
# class BART_BiLSTM(nn.Module):
#     def __init__(self, hidden_dim=256, dropout_prob=0.3, decoder_embedding_size=1024):
#         super(BART_BiLSTM, self).__init__()
#         self.hidden_dim = hidden_dim

#         # Since the input feature dimension is 1 (after reshaping), set d_model to 1
#         self.d_model = 1

#         # Transformer Encoder Layer with d_model=1
#         self.additional_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model, nhead = 8, dim_feedforward=decoder_embedding_size, batch_first=True)
        
#         self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)

#         # Linear layer to map from d_model to decoder_embedding_size
#         self.fc1 = nn.Linear(self.d_model, decoder_embedding_size)

#         # Load BART model
#         self.bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=decoder_embedding_size,
#             hidden_size=hidden_dim,
#             num_layers=1,
#             bidirectional=True,
#             batch_first=True
#         )

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout_prob)

#         # Fully Connected Layer
#         self.fc = nn.Linear(hidden_dim * 2, 1)

#     def forward(self, inputs_embeds, attention_mask, target_ids_batch_converted, input_masks_invert):
#         # Ensure inputs_embeds is of shape [batch_size, seq_len, d_model]
#         # If inputs_embeds has shape [batch_size, seq_len], add a dimension at the end
#         inputs_embeds = inputs_embeds.unsqueeze(-1)  # Now shape is [batch_size, seq_len, 1]

#         # Transformer Encoder
#         encoded_embedding = self.additional_encoder(inputs_embeds, src_key_padding_mask=input_masks_invert)

#         # Linear layer and activation
#         encoded_embedding = F.relu(self.fc1(encoded_embedding))

#         # BART model
#         out = self.bart(
#             inputs_embeds=encoded_embedding,
#             attention_mask=attention_mask,
#             return_dict=True,
#             labels=target_ids_batch_converted,
#             output_hidden_states=True
#         )

#         # Sum of the last 4 encoder layers
#         encoder_hidden_states = out.encoder_hidden_states
#         last_4_layers = encoder_hidden_states[-4:]
#         output = sum(last_4_layers)

#         # LSTM input
#         lstm_out, _ = self.lstm(output)

#         # Concatenate outputs of bidirectional LSTM
#         lstm_out_forward = lstm_out[:, -1, :self.hidden_dim]
#         lstm_out_backward = lstm_out[:, 0, self.hidden_dim:]
#         lstm_out_combined = torch.cat((lstm_out_forward, lstm_out_backward), dim=1)

#         # Apply Dropout
#         lstm_out_combined = self.dropout(lstm_out_combined)

#         # Fully Connected Layer to get predictions
#         logits = self.fc(lstm_out_combined)
#         output = torch.sigmoid(logits)

#         return output


""" main architecture for open vocabulary EEG-To-Text decoding"""
class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self,input_embeddings_batch,  input_masks_invert):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""

        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

        # encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        return encoded_embedding

    @torch.no_grad()
    def generate(
            self,
            input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
            generation_config = None,
            logits_processor = None,
            stopping_criteria = None,
            prefix_allowed_tokens_fn= None,
            synced_gpus= None,
            assistant_model = None,
            streamer= None,
            negative_prompt_ids= None,
            negative_prompt_attention_mask = None,
            **kwargs,
    ):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        output=self.pretrained.generate(
            inputs_embeds = encoded_embedding,
            attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
            labels = target_ids_batch_converted,
            return_dict = True,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs,)

        return output

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
        # print(f'forward:{input_embeddings_batch.shape,input_masks_batch.shape,input_masks_invert.shape,target_ids_batch_converted.shape,encoded_embedding.shape}')
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
                              return_dict = True, labels = target_ids_batch_converted)
        
        return out