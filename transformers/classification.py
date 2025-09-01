# import former
# from former import util
# from former.util import d, here
import util
from util import d, here

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import random, tqdm, sys, math
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter


import re, collections
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from classification_transformer import Transformer

# Used for converting between nats and bits
LOG2E = math.log(2)
NUM_CLS = 2
_WORD_RE = re.compile(r"\w+|\S")  # simple whitespace+punct tokenizer


def simple_tokenize(text):
    """Simple regex-based tokenizer"""
    return _WORD_RE.findall(text.lower())


def build_vocab(texts, vocab_size):
    """Build vocabulary from texts"""
    counter = collections.Counter()
    for text in texts:
        tokens = simple_tokenize(text)
        counter.update(tokens)
    
    # Special tokens
    vocab = {'<pad>': 0, '<unk>': 1}
    
    # Add most common words
    for word, _ in counter.most_common(vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab


def text_to_ids(text, vocab, max_length):
    """Convert text to token IDs with padding/truncation"""
    tokens = simple_tokenize(text)
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    
    # Truncate or pad
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        ids.extend([vocab['<pad>']] * (max_length - len(ids)))
    
    return torch.tensor(ids, dtype=torch.long)


def get_loaders(batch_size, vocab_size, max_len, final=False):
    """
    Load IMDB dataset using Hugging Face datasets and create data loaders
    
    Returns:
        train_loader, test_loader, vocab, pad_idx, max_length
    """
    # Load IMDB dataset
    dataset = load_dataset("imdb")
    
    if final:
        train_data = dataset['train']
        test_data = dataset['test']
    else:
        # Use train/validation split for development
        train_data = dataset['train'].train_test_split(test_size=0.2, seed=42)
        train_data, test_data = train_data['train'], train_data['test']
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    vocab = build_vocab(train_data['text'], vocab_size)
    pad_idx = vocab['<pad>']
    
    # Determine max length if not specified
    if max_len <= 0:
        lengths = [len(simple_tokenize(text)) for text in train_data['text'][:1000]]  # Sample for efficiency
        max_len = min(int(np.percentile(lengths, 95)), 512)  # Use 95th percentile, cap at 512
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Max sequence length: {max_len}")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    # Convert datasets to tensors
    def process_dataset(data):
        texts = []
        labels = []
        for item in data:
            text_ids = text_to_ids(item['text'], vocab, max_len)
            texts.append(text_ids)
            labels.append(item['label'])  # IMDB labels are already 0/1
        
        return torch.stack(texts), torch.tensor(labels, dtype=torch.long)
    
    print("Processing training data...")
    train_texts, train_labels = process_dataset(train_data)
    
    print("Processing test data...")
    test_texts, test_labels = process_dataset(test_data)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(train_texts, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_texts, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, vocab, pad_idx, max_len


def set_seed(seed: int):
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def go(args):
    """
    Creates and trains a basic transformer for the IMDB sentiment classification task.
    """
    tbw = SummaryWriter(log_dir=args.tb_dir) # TensorBoard writer for logging

    train_loader, test_loader, vocab, pad_idx, mx = get_loaders(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        max_len=args.max_length,
        final=args.final
    )

    print(f'- nr. of training batches {len(train_loader)}')
    print(f'- nr. of {"test" if args.final else "validation"} batches {len(test_loader)}')
    print(f'Using max length of {mx} for sequences')

    # ---- Model ----

    num_tokens = len(vocab)
    # Create the transformer model
    model = Transformer(embedding_dim=args.embedding_size, heads=args.num_heads, depth=args.depth,
                        seq_length=mx, num_tokens=num_tokens, num_classes=NUM_CLS) # max_pool=args.max_pool)
    
    device = util.d()
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.cuda()


    # ---- Optimizer and Scheduler ----
    opt = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)  # Optimizer for training

    # Learning rate scheduler: the lambda function scales the learning rate linearly from 0 to 1 over the warmup period.
    # After the warmup period, the learning rate remains constant. This helps stabilize training in the early stages.
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (args.lr_warmup / args.batch_size), 1.0))  # Learning rate scheduler

    # Training Loop
    seen = 0
    for e in range(args.num_epochs):
        print(f'Epoch {e}')

        model.train(True)  # Set the model to training mode

        for inputs, labels in tqdm.tqdm(train_loader, desc='Training', file=sys.stdout):
            inputs = inputs.to(device)  # Move inputs to the device
            labels = labels.to(device)  # Move labels to the device

            opt.zero_grad()  # Zero the gradients before the backward pass
            
            out = model(inputs) # (b, c) - Forward pass through the model
            loss = F.nll_loss(out, labels)  # Compute the negative log likelihood loss
            
            loss.backward()
            if args.gradient_clipping > 0.0:
                # Clip gradients to prevent exploding gradients
                # - If the total gradient vector has a length > 1, we clip it back down to 1.
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            opt.step()  # Update model parameters
            sch.step()

            seen += inputs.size(0)  # Update the number of seen examples
            tbw.add_scalar('classification/train_loss', float(loss.item()), seen)


        with torch.no_grad():

            model.train(False)  # Set the model to evaluation mode
            tot, cor = 0.0, 0.0  # Initialize total and correct counts

            for inputs, labels in test_loader:
                input = inputs.to(device)  # Move inputs to the device
                labels = labels.to(device)  # Move labels to the device

                preds = model(input).argmax(dim=1) # Get the predicted class labels
                tot += float(input.size(0))  # Update total count
                cor += float((preds == labels).sum().item()) # Update correct count
            acc = cor / tot  # Compute accuracy
            print(f'-- {"test" if args.final else "validation"} accuracy: {acc:.3f}')
        tbw.add_scalar('classification/val-acc' if not args.final else 'classification/test-acc', acc, e)
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

    parser.add_argument("-l", "--learning-rate",
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
