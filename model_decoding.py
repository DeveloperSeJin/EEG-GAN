import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartModel
import torch

class BartDiscriminator(nn.Module):
    def __init__(self):
        super(BartDiscriminator, self).__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-large')
        self.embedding_layer = self.bart.shared
        self.classifier = nn.Linear(self.bart.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input):
        dense_embedding = input @ self.embedding_layer.weight

        # BartModel 대신 Bart의 인코더만 사용
        outputs = self.bart.encoder(inputs_embeds=dense_embedding)
        
        # 첫 번째 토큰의 출력을 사용하여 분류
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)
        probabilities = self.sigmoid(logits).squeeze(-1)

        return probabilities

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.bart = BartModel.from_pretrained('facebook/bart-large')
        self.embedding_layer = self.bart.shared
        self.classifier = nn.Linear(self.bart.config.hidden_size, 1)
    
    def forward(self, input):
        dense_embedding = input @ self.embedding_layer.weight

        # BartModel 대신 Bart의 인코더만 사용
        outputs = self.bart.encoder(inputs_embeds=dense_embedding)
        
        # 첫 번째 토큰의 출력을 사용하여 분류
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        logits = self.classifier(pooled_output)

        return logits

class RNNDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_prob=0.3):
        super(RNNDiscriminator, self).__init__()

        self.embedding = nn.Linear(vocab_size, embedding_dim)


        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Convolutional layers with different kernel sizes to capture varying n-gram features
        # self.convs = nn.ModuleList([
        #     nn.Conv2d(1, num_filters, (k, vocab_size)) for k in kernel_sizes
        # ])
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        """
        x: [batch_size, seq_len, vocab_size] 형태의 one-hot 인코딩된 입력
        """
        # Add channel dimension for Conv2d
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        logit = self.fc(out).squeeze(1)  # [batch_size]
        # Apply sigmoid to get probability (0-1)
        output = torch.sigmoid(logit)
        
        return output


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