import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(torch.nn.Module):
    def __init__(self, k, heads=1, mask=False):
        super().__init__()
        assert k % heads == 0, "k must be divisible by heads"
        self.k, self.heads = k, heads

        # These compute the queries, keys and values for all
        # heads
        self.W_q = torch.nn.Linear(k, k, bias=False)
        self.W_k = torch.nn.Linear(k, k, bias=False)
        self.W_v = torch.nn.Linear(k, k, bias=False)

        # Applied after the multi-head self-attention operation
        self.unify_heads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # to divide queries, keys, values into chunks
        s = k//h

        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # fold heads into the batch dimension
        # Transpose to get the shape (b, h, t, s), then view to (b*h, t, s)

        queries = queries.transpose(1, 2).contiguous().view(b*h, t, s)
        keys = keys.transpose(1, 2).contiguous().view(b*h, t, s)
        values = values.transpose(1, 2).contiguous().view(b*h, t, s)

        # when we compute wij = q_i * k_j^T, we get a single number for each pair of i, j
        # so we need to compute the dot product for all pairs of queries and keys
        # this gives us a matrix of shape (b*h, t, t) where each row is a query and each column is a key

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b*h, t, t)
        # -- dot has size (b*h, t, t) containing raw weights
        dot = dot / (s ** 0.5)  # Scale by sqrt(d_k)

        # normalize
        dot = F.softmax(dot, dim=2)
        # - dot now contains row-wise normalized weights
        
        # apply the self attention to the values
        # dot has size (b*h, t, t) and values has size (b*h, t, s)
        out = torch.bmm(dot, values).view(b, h, t, s) # (b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, s*h)  # (b, t, k)

        # unify heads
        # out has size (b, t, k) where k = s * h
        # we need to unify the heads into a single dimension
        # this is done by a linear layer that maps (b, t, k) to (b, t, k)
        # where k is the original embedding dimension
        out = self.unify_heads(out)  # (b, t, k)
        
        return out  # (b, t, k) - the output of the self-attention mechanism
    

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.self_attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        # Feed-forward network, expansion should always be bigger than the input/output layers.
        self.ff = nn.Sequential(
            nn.Linear(k, 4*k),  # Feed-forward layer with expansion (4 times the input size, arbitrary choice)
            nn.ReLU(),
            nn.Linear(4*k, k)  # Output layer
        )
    
    def forward(self, x):
        attended = self.self_attention(x)
        x = self.norm1(attended + x)  # Add & Norm
        feedforward = self.ff(x)
        x = self.norm2(feedforward + x)  # Add & Norm
        return x