import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from io import open
from collections import Counter
from functools import partial
import unicodedata
import re
from torch.autograd import Variable
from gensim.models import KeyedVectors
import random
import time
from datetime import datetime
import pickle as pkl
import string
import os
from os import listdir 
from ast import literal_eval


### Assign indices to reserved tokens

RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}

### Data Processing Helper Functions 

def text2tokens(raw_text_fp, lang_type): 
    """ Takes filepath to raw text and outputs a list of lists, each representing a sentence of words (tokens) """
    with open(raw_text_fp) as f:
        tokens_data = [line.lower().split() for line in f.readlines()]
        if lang_type == 'source': 
            tokens_data = [datum + ['<EOS>'] for datum in tokens_data]
        elif lang_type == 'target': 
            tokens_data = [['<SOS>'] + datum + ['<EOS>'] for datum in tokens_data]
    return tokens_data 

def load_word2vec(lang): 
    """ Loads pretrained vectors for a given language """
    filepath = "data/pretrained_word2vec/wiki.zh.vec".format(lang)
    word2vec = KeyedVectors.load_word2vec_format(filepath)
    return word2vec

def build_vocab(token_lists, max_vocab_size, word2vec): 
    # UPDATE 11/28: take the most frequently occuring N words even if it doesn't exist in word2vec
    """ Takes lists of tokens (representing sentences of words), max_vocab_size, word2vec model and returns: 
        - id2token: list of tokens, where id2token[i] returns token that corresponds to i-th token 
        - token2id: dictionary where keys represent tokens and corresponding values represent their indices
        Note that the vocab will comprise N=max_vocab_size-len(RESERVED_TOKENS) tokens that are in word2vec model 
    """
    num_vocab = max_vocab_size - len(RESERVED_TOKENS)
    all_tokens = [token for sublist in token_lists for token in sublist]
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(num_vocab))
    id2token = sorted(RESERVED_TOKENS, key=RESERVED_TOKENS.get) + list(vocab)
    token2id = dict(zip(id2token, range(max_vocab_size)))
    
    # check out how many words are in word2vec vs. not 
    not_in_word2vec = [1 for token in token2id if token not in word2vec]
    pct_of_corpus = 100 * sum([token_counter[token] for token in token_counter if token not in word2vec]) / len(all_tokens)
    
    print("A vocabulary of {} is generated from a set of {} unique tokens.".format(len(token2id), len(token_counter)))
    print("{} vocab tokens are not in word2vec, comprising {:.1f}% of entire corpus.".format(len(not_in_word2vec), pct_of_corpus))
    
    return token2id, id2token 

def tokens2indices(tokens_data, token2id): 
    """ Takes tokenized data and token2id dictionary and returns indexed data """
    indices_data = [] 
    for datum in tokens_data: 
        indices_datum = [token2id[token] if token in token2id else RESERVED_TOKENS['<UNK>'] for token in datum ]
        indices_data.append(indices_datum)    
    return indices_data

def get_filepath(split, src_lang, targ_lang, lang_type): 
    """ Locates data filepath given data split type (train/dev/test), translation pairs (src_lang -> targ_lang), 
        and the language type (source or target)
    """
    folder_name = "data/iwslt-{}-{}/".format(src_lang, targ_lang)
    if lang_type == 'source': 
        file_name = "{}.tok.{}".format(split, src_lang)
    elif lang_type == 'target': 
        file_name = "{}.tok.{}".format(split, targ_lang)
    return folder_name + file_name 

def get_filepaths(src_lang, targ_lang): 
    """ Takes language names to be translated from and to (in_lang and out_lang respectively) as inputs, 
        returns a nested dictionary containing the filepaths for input/output data for train/dev/test sets  
    """
    fps = {} 
    
    # store language names 
    fps['languages'] = {} 
    fps['languages']['source'] = src_lang
    fps['languages']['target'] = targ_lang 
    
    # store filepaths 
    for split in ['train', 'dev', 'test']: 
        fps[split] = {} 
        for lang_type in ['source', 'target']: 
            fps[split][lang_type] = {} 
            fps[split][lang_type]['filepath'] = get_filepath(split, src_lang, targ_lang, lang_type)
            
    return fps 

def generate_vocab(src_lang, targ_lang, src_vocab_size, targ_vocab_size):
    # UPDATE 11/28: take the most frequently occuring N words even if it doesn't exist in word2vec
    # UPDATE 11/30: fixed bug in get_filepath; previously used global variables  
    """ Outputs a nested dictionary containing token2id, id2token, and word embeddings 
    for source and target lang's vocab """
    
    vocab = {} 
    for lang, vocab_size in zip([src_lang, targ_lang], [src_vocab_size, targ_vocab_size]): 
        
        # load train data 
        train_data_fp = get_filepath(split='train', src_lang=src_lang, targ_lang=targ_lang, 
                                     lang_type='target' if lang == 'en' else 'source')
        with open(train_data_fp) as f:
            train_tokens = [line.lower().split() for line in f.readlines()]        
        
        # load word embeddings, generate token2id and id2token 
        word2vec_full = load_word2vec(lang)
        token2id, id2token = build_vocab(train_tokens, vocab_size, word2vec_full) 
        word2vec_reduced = {word: word2vec_full[word] for word in token2id if word in word2vec_full} 
        
        # store token2id, id2token, and word embeddings as a dict in nested dict lang 
        vocab[lang] = {'token2id': token2id, 'id2token': id2token, 'word2vec': word2vec_reduced}
        
    return vocab 

def process_data(src_lang, targ_lang, vocab, sample_limit=None): 
    # UPDATE 11/27: added sample_limit parameter to output only a subset of sentences 
    """ Takes source language and target language names and respective max vocab sizes as inputs 
        and returns as a nested dictionary containing: 
        - train_indices, val_indices, test_indices (as lists of source-target tuples)
        - train_tokens, val_tokens, test_tokens (as lists of source-target tuples)
        - source language's token2id and id2token 
        - target language's token2id and id2token
    """
    
    # get filepaths 
    data = get_filepaths(src_lang, targ_lang)
    
    # loop through each file, read in text, convert to tokens, then to indices 
    for split in ['train', 'dev', 'test']: 
        for lang_type in ['source', 'target']: 
            # read in tokens 
            tokens = text2tokens(data[split][lang_type]['filepath'], lang_type)
            if sample_limit is not None: 
                tokens = tokens[:sample_limit]
            # convert tokens to indices 
            indices = tokens2indices(tokens, vocab[data['languages'][lang_type]]['token2id'])
            # save to dictionary 
            data[split][lang_type]['tokens'] = tokens
            data[split][lang_type]['indices'] = indices
            
    return data

### Create PyTorch Data Loaders 

class TranslationDataset(Dataset): 
    """ 
    Class that represents a train/validation/test/dataset that's readable for Pytorch. 
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, src_indices, targ_indices, src_max_sentence_len, targ_max_sentence_len):
        """ 
        Initialize dataset by passing in a list of input indices and a list of output indices 
        """
        self.src_indices = src_indices
        self.targ_indices = targ_indices
        self.src_max_sentence_len = src_max_sentence_len
        self.targ_max_sentence_len = targ_max_sentence_len
        assert (len(self.src_indices) == len(self.targ_indices))
        
    def __len__(self): 
        return len(self.src_indices)
    
    def __getitem__(self, key): 
        """ 
        Triggered when dataset[i] is called, outputs lists of input and output indices, as well as their 
        respective lengths
        """
        src_idx = self.src_indices[key][:self.src_max_sentence_len]
        src_len = len(src_idx)
        targ_idx = self.targ_indices[key][:self.targ_max_sentence_len]
        targ_len = len(targ_idx)
        return [src_idx, targ_idx, src_len, targ_len]
    
def collate_func(src_max_sentence_len, targ_max_sentence_len, batch): 
    """ Customized function for DataLoader that dynamically pads the batch so that all data have the same length"""
    
    src_idxs = [] 
    targ_idxs = [] 
    src_lens = [] 
    targ_lens = [] 
    
    for datum in batch: 
        # append original lengths of sequences 
        src_lens.append(datum[2]) 
        targ_lens.append(datum[3])
        
        # pad sequences before appending 
        src_idx_padded = np.pad(array=np.array(datum[0]), pad_width = ((0, src_max_sentence_len - datum[2])), 
                                mode='constant', constant_values=RESERVED_TOKENS['<PAD>'])
        targ_idx_padded = np.pad(array=np.array(datum[1]), pad_width = ((0, targ_max_sentence_len - datum[3])),
                                 mode='constant', constant_values=RESERVED_TOKENS['<PAD>'])
        src_idxs.append(src_idx_padded)
        targ_idxs.append(targ_idx_padded)
    
    return [torch.from_numpy(np.array(src_idxs)), torch.from_numpy(np.array(targ_idxs)), 
            torch.LongTensor(src_lens), torch.LongTensor(targ_lens)]

def create_dataloaders(processed_data, src_max_sentence_len, targ_max_sentence_len, batch_size): 
    """ Takes processed_data as dictionary output from process_data func, maximum sentence lengths, 
        and outputs train_loader, dev_loader, and test_loaders 
        UPDATE 11/30: output loader dict instead to easily pass into train function 
    """
    loaders = {} 
    for split in ['train', 'dev', 'test']: 
        dataset = TranslationDataset(processed_data[split]['source']['indices'], processed_data[split]['target']['indices'], 
                                     src_max_sentence_len, targ_max_sentence_len)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                    collate_fn=partial(collate_func, src_max_sentence_len, targ_max_sentence_len))
    return loaders 

