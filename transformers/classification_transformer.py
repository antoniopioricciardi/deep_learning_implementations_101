import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, embedding_dim, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, embedding_dim)
        self.pos_emb = nn.Embedding(seq_length, embedding_dim)
        self.max_pool = max_pool

        # The sequence of transformer blocks that does all the
		# heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=embedding_dim, heads=heads))
        # Create a sequential container for the transformer blocks
        # This allows us to easily apply all blocks in sequence
        self.tblocks = nn.Sequential(*tblocks)

        # The final linear layer that maps the output of the transformer
        # to the number of classes for classification (class logits)
        self.toprobs = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)  # (b, t, k)
        b, t, k = tokens.size()

        # generate positional embeddings
        pos = torch.arange(t)
        # [None, :, : ] adds a new dimension for batch size (t, k) -> (1, t, k)
        # expand replicates the positional embeddings for each batch, (1, t, k) -> (b, t, k)
        pos = self.pos_emb(pos)[None, :, :].expand(b, t, k) # (b, t, k)

        x = tokens + pos  # (b, t, k) - combine token and positional embeddings

        x = self.tblocks(x)  # (b, t, k) - pass through transformer blocks

        # Average or max pooling over the sequence length
        # If max_pool is True, we take the max over the sequence length dimension
        # Otherwise, we take the mean over the sequence length dimension
        # This reduces the sequence length dimension (t) to a single value per batch (b)
        # (b, k) - max or average pooling over the sequence length
        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # (b, k) - max or average pooling over the sequence length

        x = self.toprobs(x)  # (b, c) - average pooling over t

        return F.log_softmax(x, dim=1)
