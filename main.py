import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
    

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        """
        Reshaping this to batch size, sequence length and head size
        """
        Q = Q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # sdp logic
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        return self.output(attn_output)
    


class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.linear2(x)
    

class BertEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, intermediate_size=3072, dropout=0.1):
        super