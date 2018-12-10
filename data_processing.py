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
from gensim.models.wrappers import FastText
import random
import time
from datetime import datetime
import pickle as pkl
import string
import os
from os import listdir 
from ast import literal_eval


RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}


def text2tokens(raw_text_fp, lang_type): 
    """ Takes filepath of raw text and outputs a list of lists, each representing a sentence of words (tokens) 
        Note that it appends to target sentences <SOS> at the start, and <EOS> at the end, but only <EOS> at the end for source sentences
    """
    with open(raw_text_fp) as f:
        tokens_data = [line.lower().split() for line in f.readlines()]
        if lang_type == 'source': 
            tokens_data = [datum + ['<EOS>'] for datum in tokens_data]
        elif lang_type == 'target': 
            tokens_data = [['<SOS>'] + datum + ['<EOS>'] for datum in tokens_data]
    return tokens_data 


def load_word2vec(lang): 
    """ Loads pretrained vectors for a given language 
        Note: if lang = vi or zh load the full model (which predicts out-of-vocab embeddings), else load simple model 
    """
    if lang == 'en': 
        word2vec = KeyedVectors.load_word2vec_format("data/pretrained_word2vec/wiki.en.vec")
    else: 
        word2vec = FastText.load_fasttext_format("data/pretrained_word2vec/wiki.{}".format(lang))
    return word2vec


def build_vocab(token_lists, max_vocab_size, word2vec): 
    """ Takes lists of tokens (representing sentences of words), max_vocab_size, word2vec model and returns: 
        - id2token: list of tokens, where id2token[i] returns token that corresponds to i-th token 
        - token2id: dictionary where keys represent tokens and corresponding values represent their indices
        Note that the vocab will comprise N=max_vocab_size-len(RESERVED_TOKENS) most frequently occuring tokens, 
        including those for which we don't have pretrained embeddings. 
    """
    num_vocab = max_vocab_size - len(RESERVED_TOKENS)
    all_tokens = [token for sublist in token_lists for token in sublist]
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(num_vocab))
    id2token = sorted(RESERVED_TOKENS, key=RESERVED_TOKENS.get) + list(vocab)
    token2id = dict(zip(id2token, range(max_vocab_size)))
    
    # check how many unique tokens + pct of corpus are represented in our vocab 
    tokens_in_vocab_pct_corpus = 100 * sum([token_counter[token] for token in vocab]) / len(all_tokens)
    print("A vocabulary of {} is generated from a set of {} unique tokens, representing {:.1f}% of entire corpus".format(
        len(vocab), len(token_counter), tokens_in_vocab_pct_corpus))

    # check how many unique tokens + pct of corpus are represented in our vocab AND have pretrained embeddings 
    tokens_in_vocab_pretrained = [token for token in vocab if token in word2vec]
    tokens_in_vocab_pretrained_pct_corpus = 100 * sum([token_counter[token] for token in tokens_in_vocab_pretrained]) / len(all_tokens)
    print("{} tokens in our vocab have pretrained embeddings, representing {:.1f}% of entire corpus".format(
        len(tokens_in_vocab_pretrained), tokens_in_vocab_pretrained_pct_corpus)) 
    
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
        e.g. to load train.tok.zh, use get_filepath(split='train', src_lang='zh', targ_lang='en', lang_type='source')
    """
    folder_name = "data/iwslt-{}-{}/".format(src_lang, targ_lang)
    if lang_type == 'source': 
        file_name = "{}.tok.{}".format(split, src_lang)
    elif lang_type == 'target': 
        file_name = "{}.tok.{}".format(split, targ_lang)
    return folder_name + file_name 


def get_filepaths(src_lang, targ_lang): 
    """ Takes language names ('vi', 'zh', 'en') to be translated from and to (in_lang and out_lang respectively) as inputs, 
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
    """ Takes source and target language names and vocab sizes, outputs a nested dictionary vocab 
        containing token2id, id2token, and word2vec for both source and target languages. 
        Note the first level of keys is lang_name (e.g. 'en'), and that of nested dictionary are token2id, id2token, and word2vec.
    """
    
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


def process_data(src_lang, targ_lang, src_max_sentence_len, targ_max_sentence_len, vocab, sample_limit=None, filter_long=True): 
    """ - Main function that takes source and target language names, vocab dict generated, 
        and an optional sample_limit representing the number of sentences to subset if necessary. 
        - Returns data as a nested dictionary containing the indices and tokens of train/dev/test data 
        for both source and target languages. 
        - Note the hierachy of data dict is: data[split][lang_type]['tokens' or 'indices'], 
        e.g. to access indices of source training data, use data['train']['source']['indices']
    """ 
    
    # get filepaths 
    data = get_filepaths(src_lang, targ_lang)
    
    # loop through each file, read in text, convert to tokens, then to indices 
    for split in ['train', 'dev', 'test']: 
        for lang_type in ['source', 'target']: 
            # read in tokens 
            data[split][lang_type]['tokens'] = text2tokens(data[split][lang_type]['filepath'], lang_type)
    
    # for training data, keep only pairs with both source and target sentences within max_sent_len 
    if filter_long: 
        original_train_size = len(data['train']['source']['tokens'])
        source_lengths = np.array([len(l) for l in data['train']['source']['tokens']])
        target_lengths = np.array([len(l) for l in data['train']['target']['tokens']])
        keep_mask = (source_lengths <= src_max_sentence_len) & (target_lengths <= targ_max_sentence_len)
        data['train']['source']['tokens'] = list(np.array(data['train']['source']['tokens'])[keep_mask])
        data['train']['target']['tokens'] = list(np.array(data['train']['target']['tokens'])[keep_mask])
        new_train_size = len(data['train']['source']['tokens']) 
        print("{} data points are removed from training data after filtering out long sentences: {} remain.".format(
            new_train_size - original_train_size, new_train_size))

    # further limit number of samples if applicable 
    if sample_limit is not None: 
        for split in ['train', 'dev', 'test']: 
            for lang_type in ['source', 'target']: 
                data[split][lang_type]['tokens'] = data[split][lang_type]['tokens'][:sample_limit]

    # convert tokens to indices 
    for split in ['train', 'dev', 'test']: 
        for lang_type in ['source', 'target']: 
            data[split][lang_type]['indices'] = tokens2indices(tokens_data=data[split][lang_type]['tokens'],  
                token2id = vocab[data['languages'][lang_type]]['token2id'])

    return data


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
        outputs a nested dictionary called 'loaders' that holds train, dev, and test loaders, 
        e.g. loaders['dev'] holds the data loader for dev/validation set 
    """
    loaders = {} 
    for split in ['train', 'dev', 'test']: 
        dataset = TranslationDataset(processed_data[split]['source']['indices'], processed_data[split]['target']['indices'], 
                                     src_max_sentence_len, targ_max_sentence_len)
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                                    collate_fn=partial(collate_func, src_max_sentence_len, targ_max_sentence_len))
    return loaders 
