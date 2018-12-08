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
RESULTS_LOG = 'experiment_results/experiment_results_log.pkl'


def filter_reserved_tokens(sentence_as_list): 
    """ Takes a list of tokens representing a sentence, returns a filtered list with <SOS>, <EOS>, <PAD> removed, 
        as well as everything after the first <EOS> removed """ 
    
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
    
    list_of_lists = tensor.numpy().astype(int).tolist()
    to_token = lambda l: [id2token[idx] for idx in l]
    corpus = [to_token(l) for l in list_of_lists] 
 
    return corpus


def calc_sentence_bleu(ref_sent, hyp_sent): 
    """ Takes a reference sentence and hypothesis sentence as respective lists of tokens, outputs their bleu score """ 

    # filters out reserved tokens 
    ref_sent, hyp_sent = filter_reserved_tokens(ref_sent), filter_reserved_tokens(hyp_sent)
    
    # compute bleu score 
    sentence_bleu = sacrebleu.sentence_bleu(hyp_sent, ref_sent)

    return sentence_bleu 


def calc_corpus_bleu(ref_list, hyp_list): 
    """ Takes a list of reference sentences and a list of hypothesis sentences, outputs the average bleu score of 
        all the sentence pairs in the batch 

        *** TODO *** 
        - Convert for loop to list comprehension 
    """ 

    total_bleu = 0 
    for ref_sent, hyp_sent in zip(ref_list, hyp_list): 
        total_bleu = total_bleu + calc_sentence_bleu(ref_sent, hyp_sent)
    avg_bleu = total_bleu / len(ref_list)
    return avg_bleu 


def evaluate(model, loader, src_id2token, targ_id2token, teacher_forcing_ratio): # previously evaluate_attn
    """ Evaluates a model given a loader, id2token dicts, and teacher_forcing_ratio. 
        Outputs avg loss, avg bleu, as well as indices and tokens representing source, reference, and model translations. 
        
        *** TODO *** 
        - Accumulate bleu score rather than hold all in memory 
        - Accumulate a sample (e.g. 1 per minibatch) of source, reference, and model translations for sampling later 
        - Package output indices and tokens as two dictionaries for neatness sake 
    """
    
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
        outputs, hypotheses, attn_weights = model(src_idxs, targ_idxs, src_lens, targ_lens, 
            teacher_forcing_ratio=teacher_forcing_ratio)
        outputs = outputs[1:].transpose(0, 1)
        targets = targ_idxs[:,1:]
        attn_weights = attn_weights[:,1:]
        outputs_for_nll = outputs.contiguous().view(-1, model.decoder.targ_vocab_size)
        targets_for_nll = targets.contiguous().view(-1)
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
    lazy_eval=True, print_intermediate=1000000, save_checkpoint=True, save_to_log=True, inspect_samples=None, print_attn=False): # previously train_and_eval_attn
    
    """ Main function to train and evaluate model: takes a model, loaders, and a bunch of parameters and 
        returns trained model along with a results dict storing epoch, train/val loss, and train/val bleu scores. 

        Note that: 
        - lazy_train = train and validate only on a single mini batch (for quick prototyping) 
        - lazy_eval = skip evaluation on train set altogether (not even the 1K proxy) 
        - print_intermediate = reports loss and bleu scores every 'print_intermediate' minibatches or end of each epoch 
        - save_checkpoint = saves model's state dict into a .pth.tar file named after model_name 
        - save_to_log = 
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
    model_name = params['model_name']
    lazy_train = params['lazy_train']
    use_attn = params['use_attn']

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
            model.train()
            optimizer.zero_grad()
            final_outputs, hypotheses, attn_weights = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=teacher_forcing_ratio) 
            # attn_weights = attn_weights[:,1:]
            final_outputs = final_outputs[1:].transpose(0, 1)
            targets = targ_idxs[:,1:]                
            outputs_for_nll = final_outputs.contiguous().view(-1, model.decoder.targ_vocab_size)
            targets_for_nll = targets.contiguous().view(-1)
            loss = criterion(outputs_for_nll, targets_for_nll)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
            optimizer.step()
            
            # evaluate and report loss and bleu scores every 1000 minibatches or end of each epoch
            if batch % print_intermediate == 0 or ((epoch==num_epochs-1) & (batch==len(train_loader_)-1)):

                print("Finished gradient updates at {}".format(time.time() - start_time))

                # evaluate every epoch 
                result = {} 
        #        result['epoch'] = epoch 
                result['epoch'] = epoch + batch / len(train_loader_) 

                # calculate metrics on validation set 
                result['val_loss'], result['val_bleu'], val_hyp_idxs, val_ref_idxs, val_source_idxs, val_hyp_tokens, val_ref_tokens, val_source_tokens, val_attn = \
                    evaluate(model, dev_loader_, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio)

                print("Evaluated on validation set at {} seconds".format(time.time() - start_time))                

                # calculate metrics on train set (or proxy thereof) only if lazy_eval not set to True 
                if not lazy_eval: 
                    result['train_loss'], result['train_bleu'], train_hyp_idxs, train_ref_idxs, train_source_idxs, train_hyp_tokens, train_ref_tokens, train_source_tokens, train_attn = \
                            evaluate(model, train_loader_proxy, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio)
                else: 
                    result['train_loss'], result['train_bleu'] = 0, 0 

                print("Evaluated on training set at {} seconds".format(time.time() - start_time))    

                results.append(result)

                print("Appended results at {} seconds".format(time.time() - start_time))

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

                    print("Inspect samples at {} seconds".format(time.time() - start_time))
                    
                if save_checkpoint: 
                    if result['val_loss'] == pd.DataFrame.from_dict(results)['val_loss'].min(): 
                        checkpoint_fp = 'model_checkpoints/{}.pth.tar'.format(model_name)
                        check_dir_exists(filename=checkpoint_fp)
                        torch.save(model.state_dict(), checkpoint_fp)
                    print("Saved checkpoint at {} seconds".format(time.time() - start_time))
 
    runtime = (time.time() - start_time) / 60 
    dt_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')               

    if save_to_log: 
        append_to_log(params, results, runtime, model_name, dt_created)
        print("Appended to log at {} seconds".format(time.time() - start_time))

    print("Experiment completed in {} minutes with {:.2f} best validation loss and {:.2f} best validation BLEU.".format(
        int(runtime), pd.DataFrame.from_dict(results)['val_loss'].min(), 
        pd.DataFrame.from_dict(results)['val_bleu'].max()))

    return model, results  


def sample_predictions(hyp_idxs, ref_idxs, source_idxs, hyp_tokens, ref_tokens, source_tokens, id2token, 
    attn, print_attn, num_samples=1, ): # previously sample_predictions_attn

    """ Sample a few source sentences, reference and model translations to review """ 

    for i in range(num_samples): 

        # randomly select and index and subset out the relevant sample information   
        rand = random.randint(0, len(hyp_idxs)-1) 
        source = ' '.join(source_tokens[rand])
        print("Source: {}".format(source))
        reference_translation = ' '.join(ref_tokens[rand]) 
        print("Reference: {}".format(reference_translation))
        model_translation = ' '.join(hyp_tokens[rand])
        print("Model: {}".format(model_translation))
        if print_attn: 
            attn = attn[rand]
            print("Attention Weights: {}".format(attn))
        print()


def check_dir_exists(filename): 
    """ Takes filename string and check whether its implied directory exists, otherwise creates it """ 

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


def summarize_results(results_log): 
    """ Summarizes results_log (list) into a dataframe, splitting hyperparameters string into columns, and reducing 
        the val_acc dict into the best validation accuracy obtained amongst all the epochs logged """
    results_df = pd.DataFrame.from_dict(results_log)
    results_df = pd.concat([results_df, results_df['hyperparams'].apply(pd.Series)], axis=1)
    results_df['val_loss'] = results_df['results'].apply(lambda d: pd.DataFrame.from_dict(d)['val_loss'].min())
    return results_df.sort_values(by='dt_created', ascending=False) 


def count_parameters(model): 
    """ Returns total and trainable parameters of a given model """ 
    all_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return all_params, trainable_params


def plot_single_learning_curve(results, figsize=(8, 5)): 
    """ Plots learning curve of a SINGLE experiment, includes both train and validation accuracy """
    results_df = pd.DataFrame.from_dict(results)
    results_df = results_df.set_index('epoch')
    results_df.plot(figsize=figsize)
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')


def plot_multiple_learning_curves(results_df, plot_variable, figsize=(8, 5), legend_loc='best'):
    """ Plots learning curves of MULTIPLE experiments, includes only validation accuracy """
    plt.figure(figsize=figsize)
    for index, row in results_df.iterrows():
        val_loss_hist = pd.DataFrame.from_dict(row['results']).set_index('epoch')['val_loss'] 
        plt.plot(val_loss_hist, label="{} ({}%)".format(row[plot_variable], val_loss_hist.max()))
    plt.legend(title=plot_variable, loc=legend_loc)    
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')


## OLD CODE # 


# def train_and_eval(model, loaders_full, loaders_minibatch, loaders_minitrain, params, vocab, 
#     lazy_eval=True, print_intermediate=True, save_checkpoint=True, save_to_log=True, inspect_samples=None): 
    
#     """ Main function to train and evaluate model: takes a model, loaders, and a bunch of parameters and 
#         returns trained model along with a results dict storing epoch, train/val loss, and train/val bleu scores. 

#         Note that: 
#         - lazy_train = train and validate only on a single mini batch (for quick prototyping) 
#         - lazy_eval = skip evaluation on train set altogether (not even the 1K proxy) 
#         - print_intermediate = reports loss and bleu scores every 1000 minibatches or end of each epoch 
#         - save_checkpoint = saves model's state dict into a .pth.tar file named after model_name 
#         - save_to_log = 
#         - inspect_samples = specify number of samples to print out every 1K batches 
#     """
    
#     start_time = time.time() 

#     # extract local variables from params 
#     learning_rate = params['learning_rate'] 
#     targ_id2token = vocab[params['targ_lang']]['id2token']
#     src_id2token = vocab[params['src_lang']]['id2token']
#     num_epochs = params['num_epochs']
#     teacher_forcing_ratio = params['teacher_forcing_ratio']
#     clip_grad_max_norm = params['clip_grad_max_norm']
#     model_name = params['model_name']
#     lazy_train = params['lazy_train']

#     # designate data loaders used to train and calculate losses 
#     if lazy_train: 
#         train_loader_ = loaders_minibatch['train'] # used to train 
#         dev_loader_ = loaders_minibatch['dev'] # used to calculate dev loss 
#         train_loader_proxy = loaders_minibatch['train'] # used to calculate train loss 
#     else: 
#         train_loader_ = loaders_full['train']
#         dev_loader_ = loaders_full['dev'] 
#         # evaluating on full training set prohibitively expensive, so use a 1K batch instead as proxy 
#         train_loader_proxy = loaders_minitrain['train'] 

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
#             final_outputs, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, teacher_forcing_ratio=teacher_forcing_ratio) 
#             final_outputs = final_outputs[1:].transpose(0, 1)
#             targets = targ_idxs[:,1:]
#             outputs_for_nll = final_outputs.contiguous().view(-1, model.decoder.targ_vocab_size)
#             targets_for_nll = targets.contiguous().view(-1)
#             loss = criterion(outputs_for_nll, targets_for_nll)
#             loss.backward()
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_max_norm)
#             optimizer.step()
            
#             # evaluate and report loss and bleu scores every 1000 minibatches or end of each epoch
#             if batch % 1000 == 0 or ((epoch==num_epochs-1) & (batch==len(train_loader_)-1)):
#                 result = {} 
#                 result['epoch'] = epoch + batch / len(train_loader_) 

#                 # calculate metrics on validation set 
#                 result['val_loss'], result['val_bleu'], val_hyp_idxs, val_ref_idxs, val_source_idxs, val_hyp_tokens, val_ref_tokens, val_source_tokens = \
#                     evaluate(model, dev_loader_, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio)
#                 # calculate metrics on train set (or proxy thereof) only if lazy_eval not set to True 
#                 if not lazy_eval: 
#                     result['train_loss'], result['train_bleu'], train_hyp_idxs, train_ref_idxs, train_source_idxs, train_hyp_tokens, train_ref_tokens, train_source_tokens = \
#                             evaluate(model, train_loader_proxy, src_id2token, targ_id2token, teacher_forcing_ratio=teacher_forcing_ratio)
#                 else: 
#                     result['train_loss'], result['train_bleu'] = 0, 0 

#                 results.append(result)
                
#                 if print_intermediate: 
#                     print('Epoch: {:.2f}, Train Loss: {:.2f}, Val Loss: {:.2f}, Train BLEU: {:.2f}, Val BLEU: {:.2f}'\
#                           .format(result['epoch'], result['train_loss'], result['val_loss'], 
#                                   result['train_bleu'], result['val_bleu']))
                    
#                 if inspect_samples is not None: 
#                     # sample predictions from training set, if available 
#                     if not lazy_eval: 
#                         print("Sampling from training predictions...")
#                         sample_predictions(train_hyp_idxs, train_ref_idxs, train_source_idxs, 
#                             train_hyp_tokens, train_ref_tokens, train_source_tokens, targ_id2token, num_samples=inspect_samples)
#                     # sample predictions from validation set 
#                     print("Sampling from val predictions...")
#                     sample_predictions(val_hyp_idxs, val_ref_idxs, val_source_idxs, 
#                         val_hyp_tokens, val_ref_tokens, val_source_tokens, targ_id2token, num_samples=inspect_samples)
                    
#                 if save_checkpoint: 
#                     if result['val_loss'] == pd.DataFrame.from_dict(results)['val_loss'].min(): 
#                         checkpoint_fp = 'model_checkpoints/{}.pth.tar'.format(model_name)
#                         check_dir_exists(filename=checkpoint_fp)
#                         torch.save(model.state_dict(), checkpoint_fp)
                
#         runtime = (time.time() - start_time) / 60 
#         dt_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

#     if save_to_log: 
#         append_to_log(params, results, runtime, model_name, dt_created)

#     print("Experiment completed in {} minutes with {:.2f} best validation loss and {:.2f} best validation BLEU.".format(
#         int(runtime), pd.DataFrame.from_dict(results)['val_loss'].min(), 
#         pd.DataFrame.from_dict(results)['val_bleu'].max()))

#     return model, results  


# def evaluate(model, loader, src_id2token, targ_id2token, teacher_forcing_ratio): 
#     """ Evaluates a model given a loader, id2token dicts, and teacher_forcing_ratio. 
#         Outputs avg loss, avg bleu, as well as indices and tokens representing source, reference, and model translations. 
        
#         *** TODO *** 
#         - Accumulate bleu score rather than hold all in memory 
#         - Accumulate a sample (e.g. 1 per minibatch) of source, reference, and model translations for sampling later 
#         - Package output indices and tokens as two dictionaries for neatness sake 
#     """
    
#     model.eval() 
#     total_loss = 0 

#     # initialize empty list to hold all source, reference and model translations 
#     reference_corpus = []
#     hypothesis_corpus = [] 
#     source_corpus = [] 
    
#     for i, (src_idxs, targ_idxs, src_lens, targ_lens) in enumerate(loader): 

#         # for each batch, compute loss and accumulate to total 
#         batch_size = src_idxs.size()[0]        
#         outputs, hypotheses = model(src_idxs, targ_idxs, src_lens, targ_lens, 
#                                     teacher_forcing_ratio=teacher_forcing_ratio)
#         outputs = outputs[1:].transpose(0, 1)
#         targets = targ_idxs[:,1:]
#         outputs_for_nll = outputs.contiguous().view(-1, model.decoder.targ_vocab_size)
#         targets_for_nll = targets.contiguous().view(-1)
#         loss = F.nll_loss(outputs_for_nll, targets_for_nll, ignore_index=RESERVED_TOKENS['<PAD>'])        
#         total_loss += loss.item()  

#         # append to lists holding corpus 
#         hypothesis_corpus.append(hypotheses)
#         reference_corpus.append(targets)
#         source_corpus.append(src_idxs)

#     # concat list of index tensors into corpus tensors (as indices), then convert to list of sentence (as tokens)
#     hyp_idxs = torch.cat(hypothesis_corpus, dim=0) 
#     ref_idxs = torch.cat(reference_corpus, dim=0)
#     source_idxs = torch.cat(source_corpus, dim=0)
#     hyp_tokens = tensor2corpus(hyp_idxs, targ_id2token)
#     ref_tokens = tensor2corpus(ref_idxs, targ_id2token)
#     source_tokens = tensor2corpus(source_idxs, src_id2token)

#     # compute evaluation metrics 
#     avg_loss = total_loss / len(loader)
#     avg_bleu = calc_corpus_bleu(ref_tokens, hyp_tokens)
    
#     return avg_loss, avg_bleu, hyp_idxs, ref_idxs, source_idxs, hyp_tokens, ref_tokens, source_tokens  


# def sample_predictions(hyp_idxs, ref_idxs, source_idxs, hyp_tokens, ref_tokens, source_tokens, id2token, num_samples=1): 

#     """ Sample a few source sentences, reference and model translations to review """ 

#     for i in range(num_samples): 

#         # randomly select and index and subset out the relevant sample information   
#         rand = random.randint(0, len(hyp_idxs)-1) 
#         source = ' '.join(source_tokens[rand])
#         reference_translation = ' '.join(ref_tokens[rand]) 
#         model_translation = ' '.join(hyp_tokens[rand])

#         print("Source: {}".format(source))
#         print("Reference: {}".format(reference_translation))
#         print("Model: {}".format(model_translation))
#         print()


