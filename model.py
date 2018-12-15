import numpy as np 
import pandas as pd 
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math 


RESERVED_TOKENS = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pretrained_emb(word2vec, token2id): 
	""" Given word2vec model and vocab's token2id, generate pretrained word embeddings for all tokens in vocab. 
		For tokens not in the word2vec model, initialize with random vectors using normal distribution 
	"""

	pretrained_emb = np.zeros((len(token2id), 300)) 
	for token in token2id: 
		try: 
			pretrained_emb[token2id[token]] = word2vec[token]
		except: 
			pretrained_emb[token2id[token]] = np.random.normal(size=(300,))
	return torch.from_numpy(pretrained_emb.astype(np.float32)).to(device)


class EncoderRNN(nn.Module): # previously EncoderSimpleRNN_Test

	""" RNN encoder. Concats bidirectional hidden/output. """ 
	
	def __init__(self, rnn_cell_type, enc_hidden_dim, num_layers, enc_dropout, src_max_sentence_len, pretrained_word2vec):
		super(EncoderRNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim 
		self.enc_dropout = enc_dropout 
		self.src_max_sentence_len = src_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True) 
		self.rnn_cell_type = rnn_cell_type 
		if self.rnn_cell_type == 'gru': 
			self.rnn = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True) 
		elif self.rnn_cell_type == 'lstm': 
			self.rnn = nn.LSTM(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True) 
	
	def forward(self, enc_input, enc_input_lens):
		batch_size = enc_input.size()[0]
		_, idx_sort = torch.sort(enc_input_lens, dim=0, descending=True)
		_, idx_unsort = torch.sort(idx_sort, dim=0)
		enc_input, enc_input_lens = enc_input.index_select(0, idx_sort), enc_input_lens.index_select(0, idx_sort)
		embedded = self.embedding(enc_input)
		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, enc_input_lens, batch_first=True)
		hidden = self.initHidden(batch_size) #.to(device)
		if self.rnn_cell_type == 'gru': 
			output, hidden = self.rnn(embedded, hidden)
		elif self.rnn_cell_type == 'lstm': 
			memory = self.initHidden(batch_size) #.to(device)
			output, (hidden, memory) = self.rnn(embedded, (hidden, memory)) 
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
														   total_length=self.src_max_sentence_len,
														   padding_value=RESERVED_TOKENS['<PAD>'])
		output = output.index_select(0, idx_unsort)
		hidden = hidden.index_select(1, idx_unsort)
#		output = output[:, :, :self.enc_hidden_dim] + output[:, :, self.enc_hidden_dim:]
		output = torch.cat([output[:, :, :self.enc_hidden_dim], output[:, :, self.enc_hidden_dim:]], dim=2)
		hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_dim)
#		hidden = hidden[:, 0, :, :].squeeze(dim=1) + hidden[:, 1, :, :].squeeze(dim=1)
		hidden = torch.cat([hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1), 
			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1)], dim=2) 
		hidden = hidden.view(self.num_layers, batch_size, 2 * self.enc_hidden_dim)

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)


class DecoderRNN(nn.Module): # previously DecoderRNNV2

	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
		Handles output from EncoderRNN, which concats bidirectional output. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True) 
		self.gru = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers) 
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size) 
		self.softmax = nn.LogSoftmax(dim=1) 

	def forward(self, dec_input, dec_hidden, enc_outputs): 
		dec_input = dec_input 
		dec_hidden = dec_hidden 
		enc_outputs = enc_outputs 
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1)	
#		context = enc_outputs[:, -1, :].unsqueeze(dim=1).transpose(0, 1) 
		context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
							 enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
		concat = torch.cat([embedded, context], 2) 
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0]))  
		return output, hidden


class EncoderDecoder(nn.Module): 

	""" Encoder-Decoder without attention """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(EncoderDecoder, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.targ_vocab_size = self.decoder.targ_vocab_size
		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len 

	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 
		
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

		attn_placeholder = Variable(torch.zeros(batch_size, self.targ_max_sentence_len, self.src_max_sentence_len))

		return dec_outputs, hypotheses.transpose(0,1), attn_placeholder 


class Attention(nn.Module): 
	
	""" Implements additive attention """ 
	
	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
		super(Attention, self).__init__() 
		self.dec_hidden_dim = dec_hidden_dim
		self.input_dim = 2 * enc_hidden_dim + self.dec_hidden_dim
		self.attn = nn.Linear(self.input_dim, self.dec_hidden_dim) 
		self.v = nn.Parameter(torch.rand(self.dec_hidden_dim))
		self.num_layers = num_layers 
		nn.init.normal_(self.v, mean=0, std=1. / math.sqrt(self.dec_hidden_dim))

	def forward(self, encoder_outputs, last_dec_hidden, src_idx): 
		time_steps = encoder_outputs.size()[1]
		batch_size = encoder_outputs.size()[0]
		v_broadcast = self.v.repeat(batch_size, 1, 1) #.to(device) # [B, 1, H]
		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1) # [B, L, H] -> [B, 1, H] -> [B, H] (take last layer)
		hidden_broadcast = last_dec_hidden.repeat(1, time_steps, 1) #.to(device) # [B, T, H]
		concat = torch.cat([encoder_outputs, hidden_broadcast], dim=2) #.to(device) # [B, T, 2H]
		energies = torch.tanh(self.attn(concat)).transpose(1, 2) # [B, T, H] -> [B, H, T]
		energies = torch.bmm(v_broadcast, energies).squeeze(1) # [B, 1, H] * [B, H, T] -> [B, 1, T] -> [B, T]
		energies.data.masked_fill_(src_idx == RESERVED_TOKENS['<PAD>'], -float('inf'))
		attn_weights = F.softmax(energies, dim=1) # [B, T]

		return attn_weights


class DotAttention(nn.Module): 
	
	""" Implements multiplicative attention """
	
	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
		super(DotAttention, self).__init__() 
		self.dec_hidden_dim = dec_hidden_dim		
		self.num_layers = num_layers 

	def forward(self, encoder_outputs, last_dec_hidden, src_idx): 
		time_steps = encoder_outputs.size()[1]
		batch_size = encoder_outputs.size()[0]
		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].view(batch_size, 1, -1) # [B, L, H] -> [B, 1, H]
		energies = torch.bmm(encoder_outputs, last_dec_hidden.transpose(1, 2)).squeeze(-1)  # [B, T, H] * [B, H, 1] -> [B, T, 1] -> [B, T]
		energies.data.masked_fill_(src_idx == RESERVED_TOKENS['<PAD>'], -float('inf'))
		attn_weights = F.softmax(energies, dim=1) # [B, T]

		return attn_weights


class DecoderAttnRNN(nn.Module):

	""" Decoder with attention """ 
	
	def __init__(self, rnn_cell_type, dec_hidden_dim, enc_hidden_dim, num_layers, dec_dropout, targ_vocab_size, 
		src_max_sentence_len, targ_max_sentence_len, attention_type, pretrained_word2vec):

		super(DecoderAttnRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.src_max_sentence_len = src_max_sentence_len
		self.targ_max_sentence_len = targ_max_sentence_len
		self.targ_vocab_size = targ_vocab_size
		self.num_layers = num_layers 
		self.rnn_cell_type = rnn_cell_type 
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True) 
		# choose attention type 
		if attention_type == 'additive': 
			self.attn = Attention(self.enc_hidden_dim, self.dec_hidden_dim, 
				num_annotations = self.src_max_sentence_len, num_layers=self.num_layers) 
		elif attention_type == 'multiplicative': 
			self.attn = DotAttention(self.enc_hidden_dim, self.dec_hidden_dim, 
			num_annotations = self.src_max_sentence_len, num_layers=self.num_layers) 
		# choose RNN cell type 
		if self.rnn_cell_type == 'gru':
			self.rnn = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout) 
		elif self.rnn_cell_type == 'lstm': 
			self.rnn = nn.LSTM(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout) 
		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size) 
		self.softmax = nn.LogSoftmax(dim=1) 

	def forward(self, dec_input, dec_hidden, enc_outputs, src_idx):
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1) # [1, B, H]
		attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden, src_idx=src_idx).unsqueeze(1) # [B, 1, T]
		context = attn_weights.bmm(enc_outputs).transpose(0, 1) # [B, 1, T] * [B, T, H] = [B, 1, H] -> [1, B, H]
		concat = torch.cat([embedded, context], 2) # [1, B, 2H] 
		if self.rnn_cell_type == 'gru':
			output, hidden = self.rnn(concat, dec_hidden) # [1, B, H], [2, B, H] 
		elif self.rnn_cell_type == 'lstm':
			output, (hidden, memory) = self.rnn(concat, (dec_hidden, dec_hidden))		
		output = self.softmax(self.out(output[0])) # [B, H] -> [B, V] 

		return output, hidden, attn_weights 


class EncoderDecoderAttn(nn.Module): 

	""" Encoder Decoder with Attention """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(EncoderDecoderAttn, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.targ_vocab_size = self.decoder.targ_vocab_size
		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len 

	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 

		batch_size = src_idx.size()[0]
		enc_outputs, enc_hidden = self.encoder(src_idx, src_lens)
		dec_hidden = enc_hidden 
		dec_outputs = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_vocab_size))
		hypotheses = Variable(torch.zeros(self.targ_max_sentence_len, batch_size))
		attn_weights_all = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_max_sentence_len))
		dec_output = targ_idx[:, 0] 

		for di in range(1, self.targ_max_sentence_len): 
			dec_output, dec_hidden, attn_weights = self.decoder(dec_output, dec_hidden, enc_outputs, src_idx) # src_idx for masking 
			dec_outputs[di] = dec_output 
			teacher_labels = targ_idx[:, di-1] 
			greedy_labels = dec_output.data.max(1)[1]
			dec_output = teacher_labels if random.random() < teacher_forcing_ratio else greedy_labels 
			hypotheses[di] = greedy_labels
			attn_weights_all[di] = attn_weights.squeeze(1)

		return dec_outputs, hypotheses.transpose(0,1), attn_weights_all.transpose(0,1)

# CNN encoder
class EncoderCNN(nn.Module):
	
	def __init__(self, pretrained_word2vec, src_max_sentence_len=10, enc_hidden_dim=512, dropout=0.1):
		super(EncoderCNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.conv1_a = nn.Conv1d(300, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.conv2_a = nn.Conv1d(enc_hidden_dim, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.dropout_val = dropout
		self.src_max_sentence_len = src_max_sentence_len
		self.linearout = nn.Linear(enc_hidden_dim,300)
		self.linear_for_hidden = nn.Linear(300 * 10, self.enc_hidden_dim)
 

		
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		embedded = self.embedding(enc_input)
		embedded = F.dropout(embedded, self.dropout_val)
		
		# 1st net
		hidden_1_a = self.conv1_a(embedded.transpose(1,2)).transpose(1,2)
		#print(hidden_1_a.shape)
		#hidden_1_a.contiguous().view(-1, hidden_1_a.size(-1))
		hidden_1_a = torch.tanh(hidden_1_a.contiguous()).view(batch_size, -1, hidden_1_a.size(-1))
		hidden_2_a = self.conv2_a(hidden_1_a.transpose(1,2)).transpose(1,2)
		hidden_2_a = torch.tanh(hidden_2_a.contiguous().view(
													batch_size, -1, hidden_2_a.size(-1)))
		#print(hidden_2_a.transpose(1,2).shape)
		hidden_2_a = self.linearout(hidden_2_a)
		#print(hidden_2_a.shape)
		dim_1_hidden = self.linear_for_hidden(hidden_2_a.view(batch_size,1, -1))
		
		return hidden_2_a, dim_1_hidden
    
class EncoderCNN2(nn.Module):
	
	def __init__(self, pretrained_word2vec, src_max_sentence_len=10, enc_hidden_dim=512, dropout=0.1):
		super(EncoderCNN2, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.conv1_a = nn.Conv1d(300*src_max_sentence_len, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.conv2_a = nn.Conv1d(enc_hidden_dim, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.dropout_val = dropout
		self.src_max_sentence_len = src_max_sentence_len
		self.linearout = nn.Linear(enc_hidden_dim,3000)
		self.linear_for_hidden = nn.Linear(3000, self.enc_hidden_dim)
 

		
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		embedded = self.embedding(enc_input)
		embedded = F.dropout(embedded, self.dropout_val)
		embedded = embedded.view(batch_size, -1, 1)
		
		# 1st net
		hidden_1_a = self.conv1_a(embedded)
		#print(hidden_1_a.shape)
		#hidden_1_a.contiguous().view(-1, hidden_1_a.size(-1))
		hidden_1_a = torch.tanh(hidden_1_a.contiguous()).view(batch_size, -1, hidden_1_a.size(-1))
		hidden_2_a = self.conv2_a(hidden_1_a)
		hidden_2_a = torch.tanh(hidden_2_a.contiguous().view(
													batch_size, -1, hidden_2_a.size(-1)))
		#print(hidden_2_a.transpose(1,2).shape)
		#print(hidden_2_a.shape)
		hidden_2_a = self.linearout(hidden_2_a.transpose(1,2)).view(batch_size, -1,self.enc_embed_dim)
		#print(hidden_2_a.shape)
		dim_1_hidden = self.linear_for_hidden(hidden_2_a.view(batch_size,1, -1))
		#print(dim_1_hidden.shape)
		#print('output {}'.format(hidden_2_a.shape))
		#print('hidden {}'.format(dim_1_hidden.shape))
		
		return hidden_2_a, dim_1_hidden
    
class Decoder_RNN_from_CNN(nn.Module):
	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
		Handles output from EncoderRNN, which concats bidirectional output. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec, batch_size):
		super(Decoder_RNN_from_CNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.batch_size = batch_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True) 
		self.gru = nn.GRU(300 + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers) 
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size) 
		self.softmax = nn.LogSoftmax(dim=1) 

	def forward(self, dec_input, context, dec_hidden, enc_outputs):  
		
		batch_size = dec_input.size()[0]
		dec_hidden = dec_hidden.view(1, batch_size, -1)
		embedded = self.embedding(dec_input).view(1, batch_size, -1)   
		#print(embedded.shape)
		#context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
		#                     enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
		context = context.view(1, batch_size, -1) 
		#print(context.shape)
		concat = torch.cat([embedded, context], 2)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0]))  
		return output, hidden
    
class CNN_RNN_EncoderDecoder(nn.Module): 

	""" Encoder-Decoder without attention """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(CNN_RNN_EncoderDecoder, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.targ_vocab_size = self.decoder.targ_vocab_size
		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len

	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 
		
		batch_size = src_idx.size()[0]
		enc_outputs, enc_hidden = self.encoder(src_idx, src_lens)
		dec_hidden = enc_hidden 
		dec_outputs = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_vocab_size))
		hypotheses = Variable(torch.zeros(self.targ_max_sentence_len, batch_size))
		dec_output = targ_idx[:, 0] 

		for di in range(1, self.targ_max_sentence_len): 
			dec_output, dec_hidden = self.decoder(dec_output, dec_hidden, dec_hidden, enc_outputs)
			dec_outputs[di] = dec_output 
			teacher_labels = targ_idx[:, di-1] 
			greedy_labels = dec_output.data.max(1)[1]
			dec_output = teacher_labels if random.random() < teacher_forcing_ratio else greedy_labels 
			hypotheses[di] = greedy_labels

		attn_placeholder = Variable(torch.zeros(batch_size, self.targ_max_sentence_len, self.src_max_sentence_len))

		return dec_outputs, hypotheses.transpose(0,1), attn_placeholder 
    
class EncoderCNN(nn.Module):
	
	def __init__(self, pretrained_word2vec, src_max_sentence_len=10, enc_hidden_dim=512, dropout=0.1):
		super(EncoderCNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.conv1_a = nn.Conv1d(300, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.conv2_a = nn.Conv1d(enc_hidden_dim, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.dropout_val = dropout
		self.src_max_sentence_len = src_max_sentence_len
		self.linearout = nn.Linear(enc_hidden_dim,300)
		self.linear_for_hidden = nn.Linear(300 * 10, self.enc_hidden_dim)
 

		
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		embedded = self.embedding(enc_input)
		embedded = F.dropout(embedded, self.dropout_val)
		
		# 1st net
		hidden_1_a = self.conv1_a(embedded.transpose(1,2)).transpose(1,2)
		#print(hidden_1_a.shape)
		#hidden_1_a.contiguous().view(-1, hidden_1_a.size(-1))
		hidden_1_a = torch.tanh(hidden_1_a.contiguous()).view(batch_size, -1, hidden_1_a.size(-1))
		hidden_2_a = self.conv2_a(hidden_1_a.transpose(1,2)).transpose(1,2)
		hidden_2_a = torch.tanh(hidden_2_a.contiguous().view(
													batch_size, -1, hidden_2_a.size(-1)))
		#print(hidden_2_a.transpose(1,2).shape)
		hidden_2_a = self.linearout(hidden_2_a)
		#print(hidden_2_a.shape)
		dim_1_hidden = self.linear_for_hidden(hidden_2_a.view(batch_size,1, -1))
		
		return hidden_2_a, dim_1_hidden

class EncoderCNN2(nn.Module):
	
	def __init__(self, pretrained_word2vec, src_max_sentence_len=10, enc_hidden_dim=512, dropout=0.1):
		super(EncoderCNN2, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.conv1_a = nn.Conv1d(300*SRC_MAX_SENTENCE_LEN, enc_hidden_dim, kernel_size=3, padding=1, stride=300).to(device)
		self.conv2_a = nn.Conv1d(enc_hidden_dim, enc_hidden_dim, kernel_size=3, padding=1, stride=300).to(device)
		self.dropout_val = dropout
		self.src_max_sentence_len = src_max_sentence_len
		self.linearout = nn.Linear(enc_hidden_dim,3000)
		self.linear_for_hidden = nn.Linear(3000, self.enc_hidden_dim)
 

		
	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		embedded = self.embedding(enc_input)
		embedded = F.dropout(embedded, self.dropout_val)
		embedded = embedded.view(batch_size, -1, 1)
		
		# 1st net
		hidden_1_a = self.conv1_a(embedded)
		#print(hidden_1_a.shape)
		#hidden_1_a.contiguous().view(-1, hidden_1_a.size(-1))
		hidden_1_a = torch.tanh(hidden_1_a.contiguous()).view(batch_size, -1, hidden_1_a.size(-1))
		hidden_2_a = self.conv2_a(hidden_1_a)
		hidden_2_a = torch.tanh(hidden_2_a.contiguous().view(
													batch_size, -1, hidden_2_a.size(-1)))
		#print(hidden_2_a.transpose(1,2).shape)
		#print(hidden_2_a.shape)
		hidden_2_a = self.linearout(hidden_2_a.transpose(1,2)).view(batch_size, -1,self.enc_embed_dim)
		#print(hidden_2_a.shape)
		dim_1_hidden = self.linear_for_hidden(hidden_2_a.view(batch_size,1, -1))
		#print(dim_1_hidden.shape)
		#print('output {}'.format(hidden_2_a.shape))
		#print('hidden {}'.format(dim_1_hidden.shape))
		
		return hidden_2_a, dim_1_hidden

class Decoder_RNN_from_CNN(nn.Module):
	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
		Handles output from EncoderRNN, which concats bidirectional output. 
	""" 

	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec, batch_size):
		super(Decoder_RNN_from_CNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.targ_vocab_size = targ_vocab_size
		self.batch_size = batch_size
		self.targ_max_sentence_len = targ_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True) 
		self.gru = nn.GRU(300 + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers) 
		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size) 
		self.softmax = nn.LogSoftmax(dim=1) 

	def forward(self, dec_input, context, dec_hidden, enc_outputs):  
		
		batch_size = dec_input.size()[0]
		dec_hidden = dec_hidden.view(1, batch_size, -1)
		embedded = self.embedding(dec_input).view(1, batch_size, -1)   
		#print(embedded.shape)
		#context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
		#                     enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
		context = context.view(1, batch_size, -1) 
		#print(context.shape)
		concat = torch.cat([embedded, context], 2)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0]))  
		return output, hidden
	
class CNN_RNN_EncoderDecoder(nn.Module): 

	""" Encoder-Decoder without attention """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(CNN_RNN_EncoderDecoder, self).__init__() 
		self.encoder = encoder 
		self.decoder = decoder 
		self.targ_vocab_size = self.decoder.targ_vocab_size
		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len

	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 
		
		batch_size = src_idx.size()[0]
		enc_outputs, enc_hidden = self.encoder(src_idx, src_lens)
		dec_hidden = enc_hidden 
		dec_outputs = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_vocab_size))
		hypotheses = Variable(torch.zeros(self.targ_max_sentence_len, batch_size))
		dec_output = targ_idx[:, 0] 

		for di in range(1, self.targ_max_sentence_len): 
			dec_output, dec_hidden = self.decoder(dec_output, dec_hidden, dec_hidden, enc_outputs)
			dec_outputs[di] = dec_output 
			teacher_labels = targ_idx[:, di-1] 
			greedy_labels = dec_output.data.max(1)[1]
			dec_output = teacher_labels if random.random() < teacher_forcing_ratio else greedy_labels 
			hypotheses[di] = greedy_labels

		attn_placeholder = Variable(torch.zeros(batch_size, self.targ_max_sentence_len, self.src_max_sentence_len))

		return dec_outputs, hypotheses.transpose(0,1), attn_placeholder 