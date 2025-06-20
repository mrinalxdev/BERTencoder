import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        # Q, K, V are (batch_size, num_heads, seq_len, d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        # attention_scores: (batch_size, num_heads, seq_len_Q, seq_len_K)

        if mask is not None:
            if mask.dim() == 2: 
                if mask.size(0) == batch_size: 
                    mask = mask.unsqueeze(1).unsqueeze(2) # -> (batch_size, 1, 1, seq_len_K)
                else: # it's a causal mask (seq_len_Q, seq_len_K)
                    mask = mask.unsqueeze(0).unsqueeze(0) # -> (1, 1, seq_len_Q, seq_len_K)
            elif mask.dim() == 3: # this could happen if mask was already (batch_size, 1, seq_len)
                                  # or if it was (batch_size, seq_len_Q, seq_len_K) for some reason
                # assuming this case implies mask needs a head dimension for broadcasting
                mask = mask.unsqueeze(1) 

            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output

    def forward(self, x, mask=None, key_value=None):
        batch_size = x.size(0)
        # x is (batch_size, seq_len, d_model)
        # If key_value is None, it's self-attention (encoder or decoder self-attention)
        # If key_value is not None, it's cross-attention (decoder cross-attention)

        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Q: (batch_size, num_heads, seq_len_Q, d_k)
        
        # K and V sequence length will be based on 'x' or 'key_value'
        # In encoder, key_value is None, so K and V from x (seq_len_K = seq_len_Q)
        # In decoder self-attention, key_value is None, so K and V from x (seq_len_K = seq_len_Q)
        # In decoder cross-attention, key_value is enc_output (seq_len_K = encoder_seq_len)
        K = self.W_k(x if key_value is None else key_value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x if key_value is None else key_value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # K, V: (batch_size, num_heads, seq_len_K, d_k)

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        self_attn_output = self.self_attention(x, self_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        cross_attn_output = self.cross_attention(x, cross_mask, key_value=enc_output)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tokens, mask=None):
        batch_size, seq_len = tokens.size()
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x * np.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, mask)
        return x

class GPTDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1):
        super(GPTDecoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, tokens, enc_output, self_mask=None, cross_mask=None):
        batch_size, seq_len = tokens.size()
        positions = torch.arange(0, seq_len, device=tokens.device).unsqueeze(0).repeat(batch_size, 1)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.dropout(x * np.sqrt(self.d_model))
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, cross_mask)
        logits = self.lm_head(x)
        return logits

class BERTGPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_enc_layers, num_dec_layers, max_seq_len, dropout=0.1):
        super(BERTGPTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, d_model, num_heads, d_ff, num_enc_layers, max_seq_len, dropout)
        self.decoder = GPTDecoder(vocab_size, d_model, num_heads, d_ff, num_dec_layers, max_seq_len, dropout)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_self_mask=None, tgt_cross_mask=None):
        enc_output = self.encoder(src_tokens, src_mask)
        logits = self.decoder(tgt_tokens, enc_output, tgt_self_mask, tgt_cross_mask)
        return logits

def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

def create_dataset(src_texts, tgt_texts, tokenizer, max_seq_len):
    src_input_ids = []
    tgt_input_ids = []
    src_attention_masks = []
    for src, tgt in zip(src_texts, tgt_texts):
        src_encoded = tokenizer.encode_plus(
            src,
            add_special_tokens=True,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        tgt_encoded = tokenizer.encode_plus(
            tgt,
            add_special_tokens=True,
            max_length=max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        src_input_ids.append(src_encoded['input_ids'])
        tgt_input_ids.append(tgt_encoded['input_ids'])
        src_attention_masks.append(src_encoded['attention_mask'])
    src_input_ids = torch.cat(src_input_ids, dim=0)
    tgt_input_ids = torch.cat(tgt_input_ids, dim=0)
    src_attention_masks = torch.cat(src_attention_masks, dim=0)
    return src_input_ids, tgt_input_ids, src_attention_masks

def train_model(model, src_input_ids, tgt_input_ids, src_attention_masks, epochs=3, lr=2e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        decoder_seq_len = tgt_input_ids[:, :-1].size(1)
        tgt_self_mask = create_causal_mask(decoder_seq_len).to(tgt_input_ids.device)
        logits = model(src_input_ids, tgt_input_ids[:, :-1], src_attention_masks, tgt_self_mask, src_attention_masks)

        # Fix: Use .reshape() for target labels
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_input_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
    return model

def generate(model, src_text, tokenizer, max_seq_len, max_gen_len=20):
    model.eval()
    encoded = tokenizer.encode_plus(
        src_text,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    src_input_ids = encoded['input_ids']
    src_attention_mask = encoded['attention_mask']
    generated = [tokenizer.cls_token_id]
    with torch.no_grad():
        for _ in range(max_gen_len):
            tgt_input_ids = torch.tensor([generated], dtype=torch.long)
            tgt_self_mask = create_causal_mask(tgt_input_ids.size(1))
            logits = model(src_input_ids, tgt_input_ids, src_attention_mask, tgt_self_mask)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated.append(next_token)
            if next_token == tokenizer.sep_token_id:
                break
    return tokenizer.decode(generated, skip_special_tokens=True)

if __name__ == "__main__":
    vocab_size = 30522
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_enc_layers = 2
    num_dec_layers = 2
    max_seq_len = 32
    dropout = 0.1
    epochs = 3
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BERTGPTModel(vocab_size, d_model, num_heads, d_ff, num_enc_layers, num_dec_layers, max_seq_len, dropout)
    model.eval()
    src_texts = [
        "I love this movie, it's amazing!",
        "This film was terrible and boring.",
        "Great acting and wonderful story.",
        "Awful plot and bad performances."
    ]
    tgt_texts = [
        "This film is fantastic and I enjoy it!",
        "The movie was dull and awful.",
        "Amazing story with excellent acting.",
        "Poor storyline and terrible acting."
    ]
    src_input_ids, tgt_input_ids, src_attention_masks = create_dataset(src_texts, tgt_texts, tokenizer, max_seq_len)
    model = train_model(model, src_input_ids, tgt_input_ids, src_attention_masks, epochs=epochs)
    src_text = "I love this movie, it's amazing!"
    generated_text = generate(model, src_text, tokenizer, max_seq_len)
    print(f"Input: {src_text}")
    print(f"Generated: {generated_text}")