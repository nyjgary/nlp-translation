import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sacrebleu
import random
import time
from datetime import datetime
import pickle as pkl
import string
import os
from os import listdir 
from ast import literal_eval
from sklearn.metrics import confusion_matrix
import matplotlib.style
import matplotlib as mpl

RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}

def filter_output_indices(list_indices): 
    # NEW 11/28
    """ Filters out any tokens predicted after <EOS>, as well as <EOS>, <SOS>, and <PAD> themselves """
    
    # drops everything after <EOS> 
    try: 
        output = list_indices[:list_indices.index(RESERVED_TOKENS['<EOS>'])]
    except: 
        output = list_indices
    # drops <SOS>, <EOS>, <PAD>  
    ignored_idx = [RESERVED_TOKENS[token] for token in ['<SOS>', '<EOS>', '<PAD>']] 
    output = [idx for idx in output if idx not in ignored_idx]
    return output 

def tensor2corpus(tensor, id2token): 
    # UPDATED 11/28: Use filter_output_indices to filter out tokens predicted after <EOS> as described above 
    """ Takes a tensor (num_sentences x max_sentence_length) representing the corpus, 
        returns its string equivalent 
    """    
    
    # convert input tensor to a list of lists 
    list_of_lists = tensor.numpy().astype(int).tolist()
    
    # filter each list using above function 
    filtered = [filter_output_indices(l) for l in list_of_lists]
    
    # use dictionary to return string equivalent 
    corpus = ' '.join([id2token[idx] for l in filtered for idx in l])
    
    return corpus

def evaluate(model, loader, id2token, teacher_forcing_ratio=0.0): 
    """ 
    Helper function that tests the model's performance on a given dataset 
    @param: loader = data loader for the dataset to test against 
    """
    
    model.eval() 
    total_loss = 0 
    reference_corpus = []
    hypothesis_corpus = [] 
    
    for i, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(loader): 
        batch_size = src_idxs.size()[0]        
        outputs, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, 
                                    teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = outputs[1:].view(-1, model.decoder.targ_vocab_size)
        targets = targ_idxs[:,1:]
        hypothesis_corpus.append(hypotheses)
        reference_corpus.append(targets)
 
        loss = F.nll_loss(outputs.view(-1, model.decoder.targ_vocab_size), targets.contiguous().view(-1), 
                          ignore_index=RESERVED_TOKENS['<PAD>'])
        total_loss += loss.item()  

    # reconstruct corpus and compute bleu score 
    hypothesis_corpus = torch.cat(hypothesis_corpus, dim=0) 
    reference_corpus = torch.cat(reference_corpus, dim=0)
    hypothesis_corpus = tensor2corpus(hypothesis_corpus, id2token)
    reference_corpus = tensor2corpus(reference_corpus, id2token)
    bleu_score = sacrebleu.corpus_bleu(hypothesis_corpus, reference_corpus).score
    
    return total_loss / len(loader), bleu_score, hypothesis_corpus

# helper functions to save results to and load results from a pkl logfile 

RESULTS_LOG = 'experiment_results/experiment_results_log.pkl'

def check_dir_exists(filename): 
    """ Helper function to check that the directory of filename exists, otherwise creates it """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    else: 
        pass 
        

def append_to_log(hyperparams, results, runtime, experiment_name, dt_created, filename=RESULTS_LOG): 
    """ Appends results and details of a single experiment to a log file """
    
    # check directory exists, else creates it 
    check_dir_exists(filename)
        
    # store experiment details in a dictionary 
    new_result = {'experiment_name': experiment_name, 'hyperparams': hyperparams, 'results': results, 
                  'runtime': runtime, 'dt_created': dt_created}
    
    # if log already exists, append to log 
    try: 
        results_log = pkl.load(open(filename, "rb"))
        results_log.append(new_result)

    # if log doesn't exists, initialize first result as the log 
    except (OSError, IOError) as e:
        results_log = [new_result]
    
    # save to pickle 
    pkl.dump(results_log, open(filename, "wb")) 


def load_experiment_log(experiment_name=None, filename=RESULTS_LOG): 
    """ Loads experiment log, with option to filter for a specific experiment_name """
    
    results_log = pkl.load(open(filename, "rb"))
    
    if experiment_name is not None: 
        results_log = [r for r in results_log if r['experiment_name'] == experiment_name]
        
    return results_log


def inspect_model(model, id2token, data_split, train_loader_, dev_loader_, batch=0, num_samples=5): 
    # NEW 11/27 
    """ Use the model and output translates for first num_samples in chosen batch in chosen loader """
    
    # set loader based on data_split choice 
    if data_split == 'train': 
        loader = train_loader_ 
    elif data_split == 'val': 
        loader = dev_loader_ 
        
    for i, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(loader):
        if i == batch: 
            src_idxs = src_idxs[:num_samples, :]
            targ_idxs = targ_idxs[:num_samples, :]
            src_lens = src_lens[:num_samples]
            targ_lens = targ_lens[:num_samples]              
            output, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=0)
            
            if data_split == 'train': 
                print("Inspecting model on training data...")
            elif data_split == 'val': 
                print("Inspecting model on validation data...")
                
            print("REFERENCE TRANSLATION: {}".format(tensor2corpus(targ_idxs, id2token)))
            print("MODEL TRANSLATION: {}".format(tensor2corpus(torch.cat([hypotheses], dim=0), id2token)))
            break 
        else: 
            pass 


# def train_and_eval(model, full_loaders, fast_loaders, id2token, learning_rate, num_epochs, 
#                    print_intermediate, save_checkpoint, model_name, lazy_eval, lazy_train, inspect): 
    
#     # UPDATED 11/27: Added options to lazy_eval (skip eval on training data), lazy_train (overfit on 1 mini-batch), 
#     # and inspect (print sentences)
#     # UPDATED 11/30: Take full_loaders and fast_loaders as local variables (in dict)
    
#     if lazy_train: 
#         train_loader_ = fast_loaders['train'] 
#         dev_loader_ = fast_loaders['dev']
#     else: 
#         train_loader_ = full_loaders['train']
#         dev_loader_ = full_loaders['dev']      
    
#     # initialize optimizer and criterion 
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.NLLLoss(ignore_index=RESERVED_TOKENS['<PAD>'])
#     results = [] 
    
#     # loop through train data in batches and train 
#     for epoch in range(num_epochs): 
#         train_loss = 0 
#         for batch, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(train_loader_):
#             model.train()
#             optimizer.zero_grad()
#             final_outputs, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=0.5) 
#             loss = criterion(final_outputs[1:].view(-1, model.decoder.targ_vocab_size), targ_idxs[:,1:].contiguous().view(-1))
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
#             optimizer.step()
            
#             if batch % 100 == 0 or ((epoch==num_epochs-1) & (batch==len(train_loader_)-1)):
#                 result = {} 
#                 result['epoch'] = epoch + batch / len(train_loader_) 
#                 result['val_loss'], result['val_bleu'], val_hypotheses = evaluate(
#                     model, dev_loader_, id2token, teacher_forcing_ratio=1)
#                 if lazy_eval: 
#                     # eval on full train set is very expensive 
#                     result['train_loss'], result['train_bleu'], train_hypotheses = 0, 0, None
#                 else: 
#                     result['train_loss'], result['train_bleu'], train_hypotheses = evaluate(
#                         model, train_loader_, id2token, teacher_forcing_ratio=1)
                
#                 results.append(result)
                
#                 if print_intermediate: 
#                     print('Epoch: {:.2f}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train BLEU: {:.2f}, Val BLEU: {:.2f}'\
#                           .format(result['epoch'], result['train_loss'], result['val_loss'], 
#                                   result['train_bleu'], result['val_bleu']))
                    
#                 if inspect: 
#                     inspect_model(model, id2token, 'train', train_loader_, dev_loader_)
#                     inspect_model(model, id2token, 'val', train_loader_, dev_loader_)
                    
#                 if save_checkpoint: 
#                     if result['val_loss'] == pd.DataFrame.from_dict(results)['val_loss'].min(): 
#                         checkpoint_fp = 'model_checkpoints/{}.pth.tar'.format(model_name)
#                         check_dir_exists(filename=checkpoint_fp)
#                         torch.save(model.state_dict(), checkpoint_fp)
                
#     return results 


def train_and_eval(model, full_loaders, fast_loaders, params, vocab, print_intermediate, save_checkpoint, lazy_eval, inspect,
                   save_to_log, print_summary): 
    
    # UPDATED 11/27: Added options to lazy_eval (skip eval on training data), lazy_train (overfit on 1 mini-batch), 
    # and inspect (print sentences)
    # UPDATED 11/30: Take full_loaders and fast_loaders as local variables (in dict)
    # UPDATED 12/1: Incorporate save_to_log and print_summary 
    
    learning_rate = params['learning_rate'] 
    id2token = vocab[params['targ_lang']]['id2token']
    num_epochs = params['num_epochs']
    teacher_forcing_ratio = params['teacher_forcing_ratio']
    clip_grad_max_norm = params['clip_grad_max_norm']
    model_name = params['model_name']
    lazy_train = params['lazy_train']

    start_time = time.time() 

    if lazy_train: 
        train_loader_ = fast_loaders['train'] 
        dev_loader_ = fast_loaders['dev']
    else: 
        train_loader_ = full_loaders['train']
        dev_loader_ = full_loaders['dev']      
    
    # initialize optimizer and criterion 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=RESERVED_TOKENS['<PAD>'])
    results = [] 
    
    # loop through train data in batches and train 
    for epoch in range(num_epochs): 
        train_loss = 0 
        for batch, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(train_loader_):
            model.train()
            optimizer.zero_grad()
            final_outputs, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=teacher_forcing_ratio) 
            loss = criterion(final_outputs[1:].view(-1, model.decoder.targ_vocab_size), targ_idxs[:,1:].contiguous().view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
            optimizer.step()
            
            if batch % 100 == 0 or ((epoch==num_epochs-1) & (batch==len(train_loader_)-1)):
                result = {} 
                result['epoch'] = epoch + batch / len(train_loader_) 
                result['val_loss'], result['val_bleu'], val_hypotheses = evaluate(
                    model, dev_loader_, id2token, teacher_forcing_ratio=1)
                if lazy_eval: 
                    # eval on full train set is very expensive 
                    result['train_loss'], result['train_bleu'], train_hypotheses = 0, 0, None
                else: 
                    result['train_loss'], result['train_bleu'], train_hypotheses = evaluate(
                        model, train_loader_, id2token, teacher_forcing_ratio=1)
                
                results.append(result)
                
                if print_intermediate: 
                    print('Epoch: {:.2f}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train BLEU: {:.2f}, Val BLEU: {:.2f}'\
                          .format(result['epoch'], result['train_loss'], result['val_loss'], 
                                  result['train_bleu'], result['val_bleu']))
                    
                if inspect: 
                    inspect_model(model, id2token, 'train', train_loader_, dev_loader_)
                    inspect_model(model, id2token, 'val', train_loader_, dev_loader_)
                    
                if save_checkpoint: 
                    if result['val_loss'] == pd.DataFrame.from_dict(results)['val_loss'].min(): 
                        checkpoint_fp = 'model_checkpoints/{}.pth.tar'.format(model_name)
                        check_dir_exists(filename=checkpoint_fp)
                        torch.save(model.state_dict(), checkpoint_fp)
                
        runtime = (time.time() - start_time) / 60 
        dt_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if save_to_log: 
        append_to_log(params, results, runtime, model_name, dt_created)

    if print_summary: 
        print("Experiment completed in {} minutes with {:.2f} best validation loss and {:.2f} best validation BLEU.".format(
            int(runtime), pd.DataFrame.from_dict(results)['val_loss'].min(), 
            pd.DataFrame.from_dict(results)['val_bleu'].max()))

    return model, results  


def run_experiment(model_type, num_epochs=10, learning_rate=0.0005, num_layers=2, enc_hidden_dim=300, 
                   dec_hidden_dim=2*300, experiment_name='NA', model_name='NA', inspect=True, lazy_eval=True, 
                   lazy_train=False, save_to_log=True, save_checkpoint=False, print_summary=True, print_intermediate=True):  
    
    # UPDATED 11/27: Added options to lazy_eval, lazy_train, and inspect
    
    """ Wraps all processing, training and evaluation steps in a function to facilitate hyperparam tuning. 
        Note that the function takes as input tokenized data rather than raw data since there's significant 
        lag time in generating tokens.  
    """
    
    start_time = time.time() 
    
    # TODO: try dropout and optimization algorithms. for now use as default: 
    optimizer = 'Adam' 
    enc_dropout = 0 
    dec_dropout = 0 
    
    # instantiate model and optimizer 
    if model_type == 'without_attention': 
        encoder = EncoderRNN(enc_hidden_dim=enc_hidden_dim, num_layers=num_layers, 
                             pretrained_word2vec=get_pretrained_emb(vocab[SRC_LANG]['word2vec'], vocab[SRC_LANG]['token2id']))
        decoder = DecoderRNN(dec_hidden_dim=dec_hidden_dim, enc_hidden_dim=enc_hidden_dim, num_layers=num_layers, 
                             pretrained_word2vec=get_pretrained_emb(vocab[TARG_LANG]['word2vec'], vocab[TARG_LANG]['token2id']))
        model = EncoderDecoder(encoder, decoder, vocab[TARG_LANG]['token2id']) 
        
    elif model_type == 'attention_bahdanau': 
        encoder = EncoderRNN(enc_hidden_dim=enc_hidden_dim, num_layers=num_layers, 
                             pretrained_word2vec=get_pretrained_emb(vocab[SRC_LANG]['word2vec'], vocab[SRC_LANG]['token2id']))
        decoder = DecoderAttnRNN(dec_hidden_dim=dec_hidden_dim, enc_hidden_dim=enc_hidden_dim, num_layers=num_layers,
                                 pretrained_word2vec=get_pretrained_emb(vocab[TARG_LANG]['word2vec'], vocab[TARG_LANG]['token2id']))
        model = EncoderDecoder(encoder, decoder, vocab[TARG_LANG]['token2id'])
        
    else: 
        raise ValueError("Invalid model_type. Must be either 'without_attention' or 'attention_bahdanau'")
        
    # train and evaluate 
    results = train_and_eval(model, id2token=vocab[TARG_LANG]['id2token'], 
                             learning_rate=learning_rate, num_epochs=num_epochs, 
                             print_intermediate=print_intermediate, save_checkpoint=save_checkpoint, 
                             model_name=model_name, lazy_eval=lazy_eval, lazy_train=lazy_train, inspect=inspect)
    
    # store, print, and save results 
    runtime = (time.time() - start_time) / 60 
    dt_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hyperparams = {'model_type': model_type, 'num_epochs': num_epochs, 'learning_rate': learning_rate, 
                   'enc_hidden_dim': enc_hidden_dim, 'dec_hidden_dim': dec_hidden_dim, 'num_layers': num_layers, 
                   'optimizer': optimizer, 'enc_dropout': enc_dropout, 'dec_dropout': dec_dropout, 
                   'batch_size': BATCH_SIZE, 'src_lang': SRC_LANG, 'targ_lang': TARG_LANG, 
                   'src_vocab_size': SRC_VOCAB_SIZE, 'targ_vocab_size': TARG_VOCAB_SIZE, 
                   'src_max_sentence_len': SRC_MAX_SENTENCE_LEN, 'targ_max_sentence_len': TARG_MAX_SENTENCE_LEN}  
        
    if save_to_log: 
        append_to_log(hyperparams, results, runtime, experiment_name, dt_created)
    if print_summary: 
        print("Experiment completed in {} minutes with {:.2f} validation loss and {:.2f} validation BLEU.".format(
            int(runtime), pd.DataFrame.from_dict(results)['val_loss'].min(), 
            pd.DataFrame.from_dict(results)['val_bleu'].max()))
        
    return results, hyperparams, runtime, model


# helper methods to summarize, evaluate, and plot results 

def summarize_results(results_log): 
    """ Summarizes results_log (list) into a dataframe, splitting hyperparameters string into columns, and reducing 
        the val_acc dict into the best validation accuracy obtained amongst all the epochs logged """
    results_df = pd.DataFrame.from_dict(results_log)
    results_df = pd.concat([results_df, results_df['hyperparams'].apply(pd.Series)], axis=1)
    results_df['val_loss'] = results_df['results'].apply(lambda d: pd.DataFrame.from_dict(d)['val_loss'].min())
    return results_df.sort_values(by='val_loss', ascending=True) 

def plot_multiple_learning_curves(results_df, plot_variable, figsize=(8, 5), legend_loc='best'):
    """ Plots learning curves of MULTIPLE experiments, includes only validation accuracy """
    plt.figure(figsize=figsize)
    for index, row in results_df.iterrows():
        val_loss_hist = pd.DataFrame.from_dict(row['results']).set_index('epoch')['val_loss'] 
        plt.plot(val_loss_hist, label="{} ({}%)".format(row[plot_variable], val_loss_hist.max()))
    plt.legend(title=plot_variable, loc=legend_loc)    
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')

def plot_single_learning_curve(results, figsize=(8, 5)): 
    """ Plots learning curve of a SINGLE experiment, includes both train and validation accuracy """
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.set_index('epoch')
    results_df.plot(figsize=figsize)
    plt.ylabel('Validation Lossy')
    plt.xlabel('Epoch')


# helper function to count parameters 
def count_parameters(model): 
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params