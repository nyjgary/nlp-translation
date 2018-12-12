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
from collections import OrderedDict


RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
RESULTS_LOG = 'experiment_results/experiment_results_log.pkl'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_reserved_tokens(sentence_as_list): 
    """ Takes a list of tokens representing a sentence, removes everything after <EOS>, 
    as well as remove reserved tokens <SOS>, <EOS>, <PAD>. Outputs filtered sentence as a string. """ 

    # drops everything after <EOS> 
    try: 
        output = sentence_as_list[:sentence_as_list.index('<EOS>')]
    except: 
        output = sentence_as_list

    # drops <SOS>, <EOS>, <PAD>  
    output = ' '.join([idx for idx in output if idx not in ['<SOS>', '<EOS>', '<PAD>']]) 

    return output 


def tensor2corpus(tensor, id2token):  
    """ Takes a tensor representing a batch of sentences (size: batch_size * max_sentence_length), and returns 
        its token equivalent (as list of tokens) """ 
    
    list_of_lists = tensor.cpu().numpy().astype(int).tolist()
    to_token = lambda l: [id2token[idx] for idx in l]
    corpus = [to_token(l) for l in list_of_lists] 
 
    return corpus


def reconstruct_corpus(token_list): 
    """ Takes a list of tokens, filter out reserved tokens, and returns a list of sentence strings """ 

    sentences = [filter_reserved_tokens(sublist) for sublist in token_list]

    return sentences  


def calc_corpus_bleu(ref_list, hyp_list): 
    """ Takes a list of reference sentences and a list of hypothesis sentences, flattens them, and outputs their corpus bleu """

    # convert ref_list and hyp_list into strings 
    hyp_stream = reconstruct_corpus(hyp_list)
    ref_streams = [reconstruct_corpus(ref_list)]
    
    # compute bleu score 
    bleu_score = sacrebleu.corpus_bleu(hyp_stream, ref_streams).score  

    return bleu_score 


def evaluate(model, loader, src_id2token, targ_id2token, teacher_forcing_ratio=1): 
    """ Evaluates a model given a loader, id2token dicts, and teacher_forcing_ratio. 
        Outputs avg loss, avg bleu, as well as indices and tokens representing source, reference, and model translations. 
    """
    
    with torch.no_grad():

        model.eval() 
        total_loss = 0 

        # initialize empty list to hold all source, reference and model translations 
        reference_corpus = []
        hypothesis_corpus = [] 
        source_corpus = [] 
        attn_weights_corpus = []
        
        for i, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(loader): 

            # for each batch, compute loss and accumulate to total 
            batch_size = src_idxs.size()[0]        
            src_idxs, targ_idxs, src_lens, targ_lens = src_idxs.to(device), targ_idxs.to(device), src_lens.to(device), targ_lens.to(device)
            outputs, hypotheses, attn_weights = model(src_idxs, targ_idxs, src_lens, targ_lens, 
                teacher_forcing_ratio=teacher_forcing_ratio)
            outputs = outputs[1:].transpose(0, 1)
            targets = targ_idxs[:,1:]
            attn_weights = attn_weights[:,1:]
            outputs_for_nll = outputs.contiguous().view(-1, model.decoder.targ_vocab_size).to(device)
            targets_for_nll = targets.contiguous().view(-1).to(device)
            loss = F.nll_loss(outputs_for_nll, targets_for_nll, ignore_index=RESERVED_TOKENS['<PAD>'])        
            total_loss += loss.item()  

            # append to lists holding corpus 
            hypothesis_corpus.append(hypotheses)
            reference_corpus.append(targets)
            source_corpus.append(src_idxs)
            attn_weights_corpus.append(attn_weights)

    # concat list of index tensors into corpus tensors (as indices), then convert to list of sentence (as tokens)
    hyp_idxs = torch.cat(hypothesis_corpus, dim=0) 
    ref_idxs = torch.cat(reference_corpus, dim=0)
    source_idxs = torch.cat(source_corpus, dim=0)
    attn = torch.cat(attn_weights_corpus, dim=0)

    hyp_tokens = tensor2corpus(hyp_idxs, targ_id2token)
    ref_tokens = tensor2corpus(ref_idxs, targ_id2token)
    source_tokens = tensor2corpus(source_idxs, src_id2token)

    # compute evaluation metrics 
    avg_loss = total_loss / len(loader)
    avg_bleu = calc_corpus_bleu(ref_tokens, hyp_tokens)
    
    return avg_loss, avg_bleu, hyp_idxs, ref_idxs, source_idxs, hyp_tokens, ref_tokens, source_tokens, attn   


def train_and_eval(model, loaders_full, loaders_minibatch, loaders_minitrain, params, vocab, 
    lazy_eval=True, print_intermediate=1000000, save_checkpoint=True, save_to_log=True, inspect_samples=None, print_attn=False): 
    
    """ Main function to train and evaluate model: takes a model, loaders, and a bunch of parameters and 
        returns trained model along with a results dict storing epoch, train/val loss, and train/val bleu scores. 

        Note that: 
        - lazy_train = train and validate only on a single mini batch (for quick prototyping) 
        - lazy_eval = skip evaluation on train set altogether (not even the 1K proxy) 
        - print_intermediate = reports loss and bleu scores every 'print_intermediate' minibatches or end of each epoch 
        - save_checkpoint = saves model's state dict into a .pth.tar file named after model_name 
        - save_to_log = saves results to log 
        - inspect_samples = specify number of samples to print out every 1K batches 
    """
    
    start_time = time.time() 

    # extract local variables from params 
    learning_rate = params['learning_rate'] 
    targ_id2token = vocab[params['targ_lang']]['id2token']
    src_id2token = vocab[params['src_lang']]['id2token']
    num_epochs = params['num_epochs']
    teacher_forcing_ratio = params['teacher_forcing_ratio']
    clip_grad_max_norm = params['clip_grad_max_norm']
    experiment_name = params['experiment_name']
    model_name = params['model_name']
    lazy_train = params['lazy_train']
    attention_type = params['attention_type']

    # designate data loaders used to train and calculate losses 
    if lazy_train: 
        train_loader_ = loaders_minibatch['train'] # used to train 
        dev_loader_ = loaders_minibatch['dev'] # used to calculate dev loss 
        train_loader_proxy = loaders_minibatch['train'] # used to calculate train loss 
    else: 
        train_loader_ = loaders_full['train']
        dev_loader_ = loaders_full['dev'] 
        # evaluating on full training set prohibitively expensive, so use a 1K batch instead as proxy 
        train_loader_proxy = loaders_minitrain['train'] 

    # initialize optimizer and criterion 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=RESERVED_TOKENS['<PAD>'])
    results = [] 
    
    # loop through train data in batches and train 
    for epoch in range(num_epochs): 
        train_loss = 0 
        for batch, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(train_loader_):
            DEBUG_START = time.time() 
            src_idxs, targ_idxs, src_lens, targ_lens = src_idxs.to(device), targ_idxs.to(device), src_lens.to(device), targ_lens.to(device)
            model.train()
            optimizer.zero_grad()
            final_outputs, hypotheses, attn_weights = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=teacher_forcing_ratio) 
            # attn_weights = attn_weights[:,1:]
            final_outputs = final_outputs[1:].transpose(0, 1)
            targets = targ_idxs[:,1:]
            outputs_for_nll = final_outputs.contiguous().view(-1, model.decoder.targ_vocab_size).to(device)
            targets_for_nll = targets.contiguous().view(-1).to(device)
            loss = criterion(outputs_for_nll, targets_for_nll)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
            optimizer.step()
            
            # evaluate and report loss and bleu scores every 'print_intermediate' minibatches or end of each epoch
            if batch % print_intermediate == 0 or ((epoch==num_epochs-1) & (batch==len(train_loader_)-1)):

                result = {} 
                result['epoch'] = epoch + batch / len(train_loader_) 

                # calculate metrics on validation set 
                result['val_loss'], result['val_bleu'], val_hyp_idxs, val_ref_idxs, val_source_idxs, val_hyp_tokens, val_ref_tokens, val_source_tokens, val_attn = \
                    evaluate(model, dev_loader_, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio)         

                # calculate metrics on train set (or proxy thereof) only if lazy_eval not set to True 
                if not lazy_eval: 
                    result['train_loss'], result['train_bleu'], train_hyp_idxs, train_ref_idxs, train_source_idxs, train_hyp_tokens, train_ref_tokens, train_source_tokens, train_attn = \
                            evaluate(model, train_loader_proxy, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio) 
                else: 
                    result['train_loss'], result['train_bleu'] = 0, 0  

                results.append(result)

                print('Epoch: {:.2f}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train BLEU: {:.2f}, Val BLEU: {:.2f}, Minutes Elapsed: {:.2f}'\
                      .format(result['epoch'], result['train_loss'], result['val_loss'], 
                              result['train_bleu'], result['val_bleu'], (time.time() - start_time) / 60 ))
                    
                if inspect_samples is not None: 
                    # sample predictions from training set, if available 
                    if not lazy_eval: 
                        print("Sampling from training predictions...")
                        sample_predictions(train_hyp_idxs, train_ref_idxs, train_source_idxs, train_hyp_tokens, train_ref_tokens, 
                            train_source_tokens, targ_id2token, train_attn, print_attn=print_attn, num_samples=inspect_samples)
                    # sample predictions from validation set 
                    print("Sampling from val predictions...")
                    sample_predictions(val_hyp_idxs, val_ref_idxs, val_source_idxs, val_hyp_tokens, val_ref_tokens, val_source_tokens, 
                        targ_id2token, val_attn, print_attn=print_attn, num_samples=inspect_samples)
                    
                if save_checkpoint: 
                    if result['val_bleu'] == pd.DataFrame.from_dict(results)['val_bleu'].min(): 
                        checkpoint_fp = 'model_checkpoints/{}.pth.tar'.format(model_name)
                        check_dir_exists(filename=checkpoint_fp)
                        torch.save(model.state_dict(), checkpoint_fp)
 
    runtime = (time.time() - start_time) / 60 
    dt_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_params, trainable_params = count_parameters(model)               

    if save_to_log: 
        append_to_log(params, results, runtime, experiment_name, model_name, dt_created, total_params, trainable_params)

    print("Model training completed in {} minutes with {:.2f} best validation loss and {:.2f} best validation BLEU.".format(
        int(runtime), pd.DataFrame.from_dict(results)['val_loss'].min(), 
        pd.DataFrame.from_dict(results)['val_bleu'].max()))

    return model, results  


def sample_predictions(hyp_idxs, ref_idxs, source_idxs, hyp_tokens, ref_tokens, source_tokens, id2token, 
    attn, print_attn, num_samples=1, ): 

    """ Sample a few source sentences, reference and model translations to review """ 

    for i in range(num_samples): 
        rand = random.randint(0, len(hyp_idxs)-1) 
        source = ' '.join(source_tokens[rand])
        print("Source: {}".format(source))
        reference_translation = ' '.join(ref_tokens[rand]) 
        print("Reference: {}".format(reference_translation))
        model_translation = ' '.join(hyp_tokens[rand])
        print("Model: {}".format(model_translation))
        if print_attn: 
            print("Attention Weights: {}".format(attn[rand]))
        print()


def check_dir_exists(filename): 
    """ Takes filename string and check whether its implied directory exists, otherwise creates it """ 

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    else: 
        pass 
        

def append_to_log(hyperparams, results, runtime, experiment_name, model_name, dt_created, total_params, trainable_params, filename=RESULTS_LOG): 
    """ Appends results and details of a single experiment to a log file """
    
    # check directory exists, else creates it 
    check_dir_exists(filename)
        
    # store experiment details in a dictionary 
    new_result = {'experiment_name': experiment_name, 'model_name': model_name, 'hyperparams': hyperparams, 
        'results': results, 'runtime': runtime, 'dt_created': dt_created, 
        'total_params': total_params, 'trainable_params': trainable_params}
    
    # if log already exists, append to log 
    try: 
        results_log = pkl.load(open(filename, "rb"))
        results_log.append(new_result)

    # if log doesn't exists, initialize first result as the log 
    except (OSError, IOError) as e:
        results_log = [new_result]
    
    # save to pickle 
    pkl.dump(results_log, open(filename, "wb")) 


def load_experiment_log(experiment_name=None, model_name=None, filename=RESULTS_LOG): 
    """ Loads experiment log, with option to filter for a specific experiment_name """
    
    results_log = pkl.load(open(filename, "rb"))
    
    if experiment_name is not None: 
        results_log = [r for r in results_log if r['experiment_name'] == experiment_name]

    if model_name is not None: 
        results_log = [r for r in results_log if r['model_name'] == model_name]

    # sort by dt_created 
    results_log = sorted(results_log, key=lambda k: k['dt_created'], reverse=True)
        
    return results_log


def summarize_results(results_log): 
    """ Summarizes results_log (list) into a dataframe, splitting hyperparameters string into columns, and reducing 
        the val_acc dict into the best validation accuracy obtained amongst all the epochs logged """
    results_df = pd.DataFrame.from_dict(results_log)
    results_df = pd.concat([results_df, results_df['hyperparams'].apply(pd.Series)], axis=1)
    results_df = results_df.loc[:, ~results_df.columns.duplicated()] # unfortunately saved model_name and experiment_name twice 
    results_df['best_val_loss'] = results_df['results'].apply(lambda d: pd.DataFrame.from_dict(d)['val_loss'].min())
    results_df['best_val_bleu'] = results_df['results'].apply(lambda d: pd.DataFrame.from_dict(d)['val_bleu'].max())
    return results_df.sort_values(by='dt_created', ascending=False) 


def count_parameters(model): 
    """ Returns total and trainable parameters of a given model """ 
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def plot_single_learning_curve(results, figsize=(14, 5)): 
    """ Plots learning curve of a SINGLE experiment """
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.set_index('epoch')
    results_df = results_df[['val_bleu', 'val_loss']]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    results_df['val_loss'].plot(ax=axes[0])
    axes[0].set_ylabel('Validation Loss')
    results_df['val_bleu'].plot(ax=axes[1])
    axes[1].set_ylabel('Validation BLEU')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')


def plot_multiple_learning_curves(results_df, plot_variable, legend_title, figsize=(14, 5)):
    """ Plots learning curves of MULTIPLE experiments, includes only validation accuracy """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for index, row in results_df.iterrows():
        val_loss_hist = pd.DataFrame.from_dict(row['results']).set_index('epoch')['val_loss'] 
        axes[0].plot(val_loss_hist, label="{} ({:.2f})".format(row[plot_variable], val_loss_hist.min()))
        val_bleu_hist = pd.DataFrame.from_dict(row['results']).set_index('epoch')['val_bleu'] 
        axes[1].plot(val_bleu_hist, label="{} ({:.2f})".format(row[plot_variable], val_bleu_hist.max()))        
    axes[0].set_ylabel('Validation Loss')
    axes[1].set_ylabel('Validation BLEU ')
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].legend(title=legend_title, loc='upper right')
    axes[1].legend(title=legend_title, loc='lower right')
