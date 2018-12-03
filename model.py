import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import random


RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_emb(word2vec, token2id): 
	""" Given word2vec model and vocab's token2id, generate pretrained word embeddings for all tokens in vocab. 
		For tokens not in the word2vec model, initialize with random vectors using normal distribution 

		*** TODO *** 
		- (Optional) Consider a better way to initialize pretrained embeddings for tokens not in word2vec 
	"""

	pretrained_emb = np.zeros((len(token2id), 300)) 
	for token in token2id: 
		try: 
			pretrained_emb[token2id[token]] = word2vec[token]
		except: 
			pretrained_emb[token2id[token]] = np.random.normal(size=(300,))
	return torch.from_numpy(pretrained_emb.astype(np.float32)).to(device)


class EncoderDecoder(nn.Module): 

	""" General purpose EncoderDecoder class that should work with most if not all encoders and decoders """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(EncoderDecoder, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.targ_vocab_size = self.decoder.targ_vocab_size
		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len 

	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 
		
		src_idx, targ_idx = src_idx.to(device), targ_idx.to(device) 
		src_lens, targ_lens = src_lens.to(device), targ_lens.to(device)
		batch_size = src_idx.size()[0]
		enc_outputs, enc_hidden = self.encoder(src_idx, src_lens)
		dec_hidden = enc_hidden 
		dec_outputs = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_vocab_size))
		hypotheses = Variable(torch.zeros(self.targ_max_sentence_len, batch_size))
		dec_output = targ_idx[:, 0] 

		for di in range(1, self.targ_max_sentence_len): 
			dec_output, dec_hidden = self.decoder(dec_output, dec_hidden, enc_outputs)
			dec_outputs[di] = dec_output 
			teacher_labels = targ_idx[:, di-1] 
			greedy_labels = dec_output.data.max(1)[1]
			dec_output = teacher_labels if random.random() < teacher_forcing_ratio else greedy_labels 
			hypotheses[di] = greedy_labels

		return dec_outputs, hypotheses.transpose(0,1)


class EncoderRNN(nn.Module):

	""" Vanilla RNN encoder, returns twice the original hidden dimension due to bidirectional 
	
		*** TODO *** 
		- Haven't retested after major bug fix. Retry later. 

	""" 

	def __init__(self, enc_hidden_dim, num_layers, src_max_sentence_len, pretrained_word2vec):
		super(EncoderRNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim 
		self.src_max_sentence_len = src_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.gru = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
						  batch_first=True, bidirectional=True).to(device)
	
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		_, idx_sort = torch.sort(enc_input_lens, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		enc_input, enc_input_lens = enc_input.index_select(0, idx_sort), enc_input_lens.index_select(0, idx_sort)
		embedded = self.embedding(enc_input)
		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, enc_input_lens, batch_first=True)
		hidden = self.initHidden(batch_size).to(device)
		output, hidden = self.gru(embedded, hidden)
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
														   total_length=self.src_max_sentence_len,
														   padding_value=RESERVED_TOKENS['<PAD>'])
		output = output.index_select(0, idx_unsort)
		hidden = hidden.index_select(1, idx_unsort).transpose(0, 1).contiguous().view(self.num_layers, batch_size, -1)

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)
	

class EncoderSimpleRNN(nn.Module):

	""" Vanilla RNN encoder. Sums the bidirectional hidden/output instead of returning twice the hidden dimension """ 
	
	def __init__(self, enc_hidden_dim, num_layers, src_max_sentence_len, pretrained_word2vec):
		super(EncoderSimpleRNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim 
		self.src_max_sentence_len = src_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.gru = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
						  batch_first=True, bidirectional=True).to(device)
	
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		_, idx_sort = torch.sort(enc_input_lens, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		enc_input, enc_input_lens = enc_input.index_select(0, idx_sort), enc_input_lens.index_select(0, idx_sort)
		embedded = self.embedding(enc_input)
		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, enc_input_lens, batch_first=True)
		hidden = self.initHidden(batch_size).to(device)
		output, hidden = self.gru(embedded, hidden)
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
														   total_length=self.src_max_sentence_len,
														   padding_value=RESERVED_TOKENS['<PAD>'])
		output = output.index_select(0, idx_unsort)
		hidden = hidden.index_select(1, idx_unsort)
		output = output[:, :, :self.enc_hidden_dim] + output[:, :, self.enc_hidden_dim:]
		hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_dim)
		hidden = hidden[:, 0, :, :].squeeze(dim=1) + hidden[:, 1, :, :].squeeze(dim=1)
		hidden = hidden.view(self.num_layers, batch_size, self.enc_hidden_dim)

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)


class DecoderRNN(nn.Module):

	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
		Handles output from EncoderRNN that returns twice the encoder hidden dimension.  

		*** TODO *** 
		- Haven't retested after major bug fix. Retry later. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.gru = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size).to(device)
		self.softmax = nn.LogSoftmax(dim=1).to(device)

	def forward(self, dec_input, dec_hidden, enc_outputs): 
		dec_input = dec_input.to(device)
		dec_hidden = dec_hidden.to(device)
		enc_outputs = enc_outputs.to(device)
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1)
		context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
							 enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
		concat = torch.cat([embedded, context], 2).to(device)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0].to(device)))    
		return output, hidden


class DecoderRNNV2(nn.Module):

	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
		This handles the output from EncoderSimpleRNN, which sums the bidrectional output. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderRNNV2, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.gru = nn.GRU(self.dec_embed_dim + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size).to(device)
		self.softmax = nn.LogSoftmax(dim=1).to(device)

	def forward(self, dec_input, dec_hidden, enc_outputs): 
		dec_input = dec_input.to(device)
		dec_hidden = dec_hidden.to(device)
		enc_outputs = enc_outputs.to(device)
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1)	
		context = enc_outputs[:, -1, :].unsqueeze(dim=1).transpose(0, 1) 
		concat = torch.cat([embedded, context], 2).to(device)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0].to(device)))    
		return output, hidden


class DecoderSimpleRNN(nn.Module):

	""" Vanilla decoder without attention, and final encoder hidden layer NOT passed to every time step of decoder 

		*** TODO *** 
		- Haven't retested after major bug fix. Retry later. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderSimpleRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.gru = nn.GRU(self.dec_embed_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size).to(device)
		self.softmax = nn.LogSoftmax(dim=1).to(device)

	def forward(self, dec_input, dec_hidden, enc_outputs): 
		dec_input = dec_input.to(device)
		dec_hidden = dec_hidden.to(device)
		enc_outputs = enc_outputs.to(device)
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1)
		dec_hidden = dec_hidden.view(self.num_layers, batch_size, self.dec_hidden_dim)
		output, hidden = self.gru(embedded, dec_hidden)
		output = self.softmax(self.out(output[0].to(device)))    

		return output, hidden
		

class Attention(nn.Module): 
	
	""" Implements the attention mechanism by Bahdanau et al. (2015) 

		*** TODO *** 
		- Haven't retested after major bug fix. Retry later. 
	"""
	
	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
		super(Attention, self).__init__() 
		self.num_annotations = num_annotations
		self.input_dim = enc_hidden_dim * 2 + dec_hidden_dim
		self.attn = nn.Linear(self.input_dim, self.num_annotations).to(device)
		self.v = nn.Parameter(torch.rand(self.num_annotations))
		self.num_layers = num_layers 
		nn.init.normal_(self.v)
		
	def forward(self, encoder_outputs, last_dec_hidden): 
		encoder_outputs, last_dec_hidden = encoder_outputs.to(device), last_dec_hidden.to(device)
		batch_size = encoder_outputs.size()[0]
		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1) 
		hidden_broadcast = last_dec_hidden.repeat(1, self.num_annotations, 1).to(device)
		v_broadcast = self.v.repeat(batch_size, 1, 1).to(device)
		concat = torch.cat([encoder_outputs, hidden_broadcast], dim=2).to(device)
		energies = v_broadcast.bmm(torch.tanh(self.attn(concat)))
		attn_weights = F.softmax(energies, dim=2).squeeze(1)
		return attn_weights

class DecoderAttnRNN(nn.Module):

	""" Decoder with attention (Bahdanau) 

		*** TODO *** 
		- Haven't retested after major bug fix. Retry later. 
	""" 
	
	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, src_max_sentence_len, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderAttnRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.src_max_sentence_len = src_max_sentence_len
		self.targ_max_sentence_len = targ_max_sentence_len
		self.targ_vocab_size = targ_vocab_size
		self.num_layers = num_layers 
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.attn = Attention(self.enc_hidden_dim, self.dec_hidden_dim, 
							  num_annotations = self.src_max_sentence_len, num_layers=self.num_layers).to(device)
		self.gru = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size).to(device)
		self.softmax = nn.LogSoftmax(dim=1).to(device)

	def forward(self, dec_input, dec_hidden, enc_outputs):
		dec_input, dec_hidden = dec_input.to(device), dec_hidden.to(device)
		enc_outputs = enc_outputs.to(device)
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1)
		attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden).unsqueeze(1)
		context = attn_weights.bmm(enc_outputs).transpose(0, 1)
		concat = torch.cat([embedded, context], 2).to(device)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0].to(device)))    
		return output, hidden


