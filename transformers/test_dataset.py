#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

from datasets import load_dataset
import torch
import re
import collections

_WORD_RE = re.compile(r"\w+|\S")

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

def test_dataset_loading():
    """Test the dataset loading functionality"""
    print("Loading IMDB dataset...")
    
    # Load just a small sample first
    dataset = load_dataset("imdb", split="train[:100]")  # Just 100 samples for testing
    
    print(f"Dataset loaded successfully! {len(dataset)} samples")
    print(f"Features: {dataset.features}")
    
    # Test tokenization and vocab building
    print("\nTesting tokenization...")
    sample_text = dataset[0]['text']
    tokens = simple_tokenize(sample_text)
    print(f"Sample text length: {len(sample_text)} chars")
    print(f"Tokens: {tokens[:10]}...")  # First 10 tokens
    print(f"Total tokens: {len(tokens)}")
    
    # Test vocabulary building
    print("\nBuilding vocabulary from sample...")
    texts = [item['text'] for item in dataset]
    vocab = build_vocab(texts, 1000)  # Small vocab for testing
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Sample vocab entries: {list(vocab.items())[:10]}")
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_dataset_loading()
