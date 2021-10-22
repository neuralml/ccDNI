#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 12:15:08 2019

@author: joe
penn treebank dataset where using BATCH_FIRST
"""
import os
import torch

from torch.utils.data import DataLoader, random_split, Dataset

############DATA###############
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

def sequify(data, seq_len):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // seq_len
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * seq_len)
    # Evenly divide the data across the sequence length batches.
    data = data.view(seq_len, -1).t().contiguous()
    return data

class langDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, tensor, seq_len):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensor = tensor
        self.seq_len = seq_len
        
    def __getitem__(self, index):
        
        if self.seq_len is not None:
            i = index * self.seq_len
            features = self.tensor[i:i + self.seq_len]
            targets = self.tensor[i+1:i + 1 + self.seq_len]
        else:
            features = self.tensor[:-1]
            targets = self.tensor[1:]
        
        #print("Features shape:", features.shape)
        return tuple((features, targets))

    def __len__(self):
        #return self.tensors[0].size(0)
        if self.seq_len is None:
            return 1
        else:
            return len(self.tensor) // self.seq_len 

def load_penn_words(data_path, seq_len, batch_size, shuffle=False):
    corpus = Corpus(data_path)
    train_raw = langDataset(corpus.train, seq_len)
    
    val_raw = langDataset(corpus.valid, None)
    test_raw = langDataset(corpus.test, None)
    
    ntokens = len(corpus.dictionary)
    
    train_data = DataLoader(
        train_raw, batch_size=batch_size, shuffle=shuffle)
    validation_data = DataLoader(
        val_raw, batch_size=batch_size, shuffle=False)
    test_data = DataLoader(
        test_raw, batch_size=batch_size, shuffle=False)    
    
    return train_data, validation_data, test_data, ntokens
    
    