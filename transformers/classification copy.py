# import former
# from former import util
# from former.util import d, here
import util
from util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

# Modern torchtext imports
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

import random, tqdm, sys, math, gzip

from transformers.classification_transformer import Transformer
# Used for converting between nats and bits
LOG2E = math.log(2)

# Modern tokenizer and vocabulary setup
tokenizer = get_tokenizer('basic_english')
NUM_CLS = 2

def build_vocab_from_iterator(data_iter, vocab_size):
    """Build vocabulary from text data iterator"""
    counter = Counter()
    for text, _ in data_iter:
        tokens = tokenizer(text.lower())
        counter.update(tokens)
    
    # Keep most common words, reserve space for <unk> and <pad>
    vocab_dict = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(vocab_size - 2):
        vocab_dict[word] = len(vocab_dict)
    
    return vocab_dict

def text_pipeline(vocab_dict, text):
    """Convert text to tensor of token indices"""
    tokens = tokenizer(text.lower())
    return [vocab_dict.get(token, vocab_dict['<unk>']) for token in tokens]

def collate_batch(batch, vocab_dict, max_length):
    """Collate function for DataLoader"""
    texts, labels = [], []
    for text, label in batch:
        processed_text = torch.tensor(text_pipeline(vocab_dict, text), dtype=torch.long)
        # Truncate or pad to max_length
        if len(processed_text) > max_length:
            processed_text = processed_text[:max_length]
        else:
            padding = torch.zeros(max_length - len(processed_text), dtype=torch.long)
            processed_text = torch.cat([processed_text, padding])
        
        texts.append(processed_text)
        labels.append(label)
    
    return torch.stack(texts), torch.tensor(labels, dtype=torch.long)


def go(args):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=args.tb_dir) # TensorBoard writer for logging

    # Load the IMDB dataset using modern torchtext API
    if args.final:
        train_dataset = IMDB(split='train')
        test_dataset = IMDB(split='test')
    else:
        # For validation, we'll use part of the training set
        full_train_dataset = IMDB(split='train')
        # Convert to list to split
        full_train_list = list(full_train_dataset)
        split_idx = int(0.8 * len(full_train_list))
        train_list = full_train_list[:split_idx]
        test_list = full_train_list[split_idx:]
        train_dataset = train_list
        test_dataset = test_list

    # Build vocabulary from training data
    vocab_dict = build_vocab_from_iterator(train_dataset, args.vocab_size)
    
    print(f'- vocabulary size: {len(vocab_dict)}')
    print(f'- nr. of training examples {len(list(train_dataset))}')
    print(f'- nr. of {"test" if args.final else "validation"} examples {len(list(test_dataset))}')

    if args.max_length < 0:
        # Find maximum sequence length in training data
        max_len = 0
        for text, _ in train_dataset:
            tokens = text_pipeline(vocab_dict, text)
            max_len = max(max_len, len(tokens))
        mx = max_len * 2  # Double the max length to account for padding
    else:
        mx = args.max_length
    print(f'Using max length of {mx} for sequences')

    # Create data loaders with custom collate function
    from functools import partial
    collate_fn = partial(collate_batch, vocab_dict=vocab_dict, max_length=mx)
    
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Create the transformer model
    model = Transformer(embedding_dim=args.embedding_size, heads=args.num_heads, depth=args.depth,
                        seq_length=mx, num_tokens=args.vocab_size, num_classes=NUM_CLS) # max_pool=args.max_pool)
    if torch.cuda.is_available():
        model = model.cuda()
    
    opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)  # Optimizer for training

    # Learning rate scheduler: the lambda function scales the learning rate linearly from 0 to 1 over the warmup period.
    # After the warmup period, the learning rate remains constant. This helps stabilize training in the early stages.
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (args.lr_warmup / args.batch_size), 1.0))  # Learning rate scheduler

    # Training Loop
    seen = 0
    for e in range(args.num_epochs):
        print(f'Epoch {e}')

        model.train(True)  # Set the model to training mode

        # file=sys.stdout means that tqdm will print to the console
        for batch in tqdm.tqdm(train_iter, desc='Training', file=sys.stdout):
            opt.zero_grad()  # Zero the gradients before the backward pass

            input, label = batch  # Get the input text and labels from the batch
            label = label - 1 # Adjust label to be zero-indexed (IMDB labels are 1,2 -> 0,1)

            if input.size(1) > mx:
                input = input[:, :mx] # Truncate the input to the maximum length if it exceeds mx
            out = model(input) # Forward pass through the model
            loss = F.nll_loss(out, label)  # Compute the negative log likelihood loss

            loss.backward()  # Backward pass to compute gradients

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if args.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

            opt.step()
            sch.step()  # Step the learning rate scheduler

            seen += input.size(0)
            tbw.add_scalar('classification/train_loss', float(loss.item()), seen)  # Log the training loss to TensorBoard

        with torch.no_grad():

            model.train(False)  # Set the model to evaluation mode
            tot, cor = 0.0, 0.0  # Initialize total and correct counts

            for batch in test_iter:
                input, label = batch  # Get input and labels from batch
                label = label - 1  # Adjust label to be zero-indexed

                if input.size(1) > mx:
                    input = input[:, :mx]
                out = model(input).argmax(dim=1) # Get the predicted class labels

                tot += float(input.size(0))  # Update total count
                cor += float((out == label).sum().item())  # Update correct count
            acc = cor / tot  # Compute accuracy
            print(f'-- {"test" if args.final else "validation"} accuracy: {acc:.3f}')
            tbw.add_scalar('classification/test-loss', float(loss.item()), e)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-e', '--num-epochs',
                        dest="num_epochs",
                        help="Number of epochs.",
                        default=80, type=int)
    
    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=4, type=int)

    parser.add_argument("-l", "--learn-rate",
                        dest="learning_rate",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb_dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("--max-pool", dest="max_pool",
                        help="Use max pooling in the final classification layer.",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-V", "--vocab-size", dest="vocab_size",
                        help="Number of words in the vocabulary.",
                        default=50_000, type=int)

    parser.add_argument("-M", "--max", dest="max_length",
                        help="Max sequence length. Longer sequences are clipped (-1 for no limit).",
                        default=512, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of self-attention layers)",
                        default=6, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=10_000, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    options = parser.parse_args()

    print('OPTIONS:', options)

    go(options)  # Start the training process
    print('Done.')
