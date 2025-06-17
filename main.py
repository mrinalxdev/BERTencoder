import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, V)
        return output, attention_probs

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention_output, attention_probs = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attention_output)
        return output, attention_probs

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
        attn_output, attention_probs = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attention_probs

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
        attention_probs_all = []
        for layer in self.layers:
            x, attention_probs = layer(x, mask)
            attention_probs_all.append(attention_probs)
        return x, attention_probs_all

def visualize_attention(attention_probs, tokens, vocab, layer_idx=0, save_path="attention_heatmap.png"):
    """
    Visualize attention weights for a single sequence in a specific layer.
    
    Args:
        attention_probs: List of tensors, each of shape [batch_size, num_heads, seq_len, seq_len]
        tokens: Tensor of shape [batch_size, seq_len]
        vocab: Dictionary mapping token IDs to token strings
        layer_idx: Layer to visualize (0-based)
        save_path: Path to save the heatmap
    """
    attn = attention_probs[layer_idx][0].detach().cpu().numpy()  # [num_heads, seq_len, seq_len]
    seq_len = attn.shape[-1]
    tokens = tokens[0].detach().cpu().numpy()  # [seq_len]
    token_labels = [vocab.get(t, "[UNK]") for t in tokens]


    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for head in range(min(attn.shape[0], 8)):  # Handle up to 8 heads
        sns.heatmap(attn[head], ax=axes[head], cmap="viridis", xticklabels=token_labels, yticklabels=token_labels)
        axes[head].set_title(f"Head {head + 1}")
        axes[head].set_xlabel("Key Tokens")
        axes[head].set_ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Attention heatmap saved to {save_path}")

if __name__ == "__main__":
    vocab_size = 100
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 2
    max_seq_len = 10
    dropout = 0.1
    batch_size = 1
    seq_len = 5

    # Simple vocabulary for visualization
    vocab = {i: f"token_{i}" for i in range(vocab_size)}
    vocab[0] = "[PAD]"
    vocab[1] = "[CLS]"
    vocab[2] = "[SEP]"

    # Initialize model
    model = BERTEncoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout)
    model.eval()

    # Dummy input (single sequence)
    tokens = torch.tensor([[1, 10, 20, 30, 2]], dtype=torch.long)  # [CLS], token_10, token_20, token_30, [SEP]
    mask = torch.ones(batch_size, seq_len, seq_len)  # All tokens attend to each other

    output, attention_probs_all = model(tokens, mask)
    print(f"Input shape: {tokens.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention probs per layer: {[probs.shape for probs in attention_probs_all]}")
    visualize_attention(attention_probs_all, tokens, vocab, layer_idx=0, save_path="attention_heatmap_layer0.png")