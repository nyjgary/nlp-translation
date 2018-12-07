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


class EncoderRNN(nn.Module): # previously EncoderSimpleRNN_Test

	""" RNN encoder. Concats bidirectional hidden/output. """ 
	
	def __init__(self, rnn_cell_type, enc_hidden_dim, num_layers, enc_dropout, src_max_sentence_len, pretrained_word2vec):
		super(EncoderRNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim 
		self.enc_dropout = enc_dropout 
		self.src_max_sentence_len = src_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.rnn_cell_type = rnn_cell_type 
		if self.rnn_cell_type == 'gru': 
			self.rnn = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True).to(device)
		elif self.rnn_cell_type == 'lstm': 
			self.rnn = nn.LSTM(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True).to(device)
	
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
#		print("hidden initialized is {}".format(hidden.size()))
		if self.rnn_cell_type == 'gru': 
			output, hidden = self.rnn(embedded, hidden)
		elif self.rnn_cell_type == 'lstm': 
			memory = self.initHidden(batch_size).to(device)
			output, (hidden, memory) = self.rnn(embedded, (hidden, memory)) 
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
														   total_length=self.src_max_sentence_len,
														   padding_value=RESERVED_TOKENS['<PAD>'])
#		print("output packed is {}".format(output.size()))
		output = output.index_select(0, idx_unsort)
		hidden = hidden.index_select(1, idx_unsort)
		# print("output left is {} output right is {}".format( output[:, :, :self.enc_hidden_dim].size(), 
		# 	output[:, :, self.enc_hidden_dim:].size()))
#		output = output[:, :, :self.enc_hidden_dim] + output[:, :, self.enc_hidden_dim:]
#		print("output left is {}".format(output[:, :, :self.enc_hidden_dim].size()))
#		print("output right is {}".format(output[:, :, self.enc_hidden_dim:].size()))
		output = torch.cat([output[:, :, :self.enc_hidden_dim], output[:, :, self.enc_hidden_dim:]], dim=2)
#		print("after cat output is {}".format(output.size()))
		hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_dim)
#		print("after view hidden is {}".format(hidden.size()))
#		print("before squeezing hidden left is {} hidden right is {}".format(
#			hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).size(), 
#			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).size()))
#		print("after squeezing hidden left is {} hidden right is {}".format(
#			hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1).size(), 
#			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1).size()))
#		hidden = hidden[:, 0, :, :].squeeze(dim=1) + hidden[:, 1, :, :].squeeze(dim=1)
#		hidden = torch.cat([hidden[:, 0, :, :].squeeze(dim=1), hidden[:, 1, :, :].squeeze(dim=1)], dim=2) 
		hidden = torch.cat([hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1), 
			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1)], dim=2) 
#		print("after cat hidden is {}".format(hidden.size()))
		hidden = hidden.view(self.num_layers, batch_size, 2 * self.enc_hidden_dim)
#		print("after view hidden is {}".format(hidden.size()))

		return output, hidden

	def initHidden(self, batch_size):
		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)


class EncoderRNN_Mask(nn.Module): # previously EncoderSimpleRNN_Test

	""" RNN encoder. Concats bidirectional hidden/output. """ 
	
	def __init__(self, rnn_cell_type, enc_hidden_dim, num_layers, enc_dropout, src_max_sentence_len, pretrained_word2vec):
		super(EncoderRNN, self).__init__()
		self.enc_embed_dim = 300
		self.enc_hidden_dim = enc_hidden_dim 
		self.enc_dropout = enc_dropout 
		self.src_max_sentence_len = src_max_sentence_len
		self.num_layers = num_layers
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.rnn_cell_type = rnn_cell_type 
		if self.rnn_cell_type == 'gru': 
			self.rnn = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True).to(device)
		elif self.rnn_cell_type == 'lstm': 
			self.rnn = nn.LSTM(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
				dropout = enc_dropout, batch_first=True, bidirectional=True).to(device)
	
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
#		print("hidden initialized is {}".format(hidden.size()))
		if self.rnn_cell_type == 'gru': 
			output, hidden = self.rnn(embedded, hidden)
		elif self.rnn_cell_type == 'lstm': 
			memory = self.initHidden(batch_size).to(device)
			output, (hidden, memory) = self.rnn(embedded, (hidden, memory)) 
		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
														   total_length=self.src_max_sentence_len,
														   padding_value=RESERVED_TOKENS['<PAD>'])
#		print("output packed is {}".format(output.size()))
		output = output.index_select(0, idx_unsort)
		hidden = hidden.index_select(1, idx_unsort)
		# print("output left is {} output right is {}".format( output[:, :, :self.enc_hidden_dim].size(), 
		# 	output[:, :, self.enc_hidden_dim:].size()))
#		output = output[:, :, :self.enc_hidden_dim] + output[:, :, self.enc_hidden_dim:]
#		print("output left is {}".format(output[:, :, :self.enc_hidden_dim].size()))
#		print("output right is {}".format(output[:, :, self.enc_hidden_dim:].size()))
		output = torch.cat([output[:, :, :self.enc_hidden_dim], output[:, :, self.enc_hidden_dim:]], dim=2)
#		print("after cat output is {}".format(output.size()))
		hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_dim)
#		print("after view hidden is {}".format(hidden.size()))
#		print("before squeezing hidden left is {} hidden right is {}".format(
#			hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).size(), 
#			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).size()))
#		print("after squeezing hidden left is {} hidden right is {}".format(
#			hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1).size(), 
#			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1).size()))
#		hidden = hidden[:, 0, :, :].squeeze(dim=1) + hidden[:, 1, :, :].squeeze(dim=1)
#		hidden = torch.cat([hidden[:, 0, :, :].squeeze(dim=1), hidden[:, 1, :, :].squeeze(dim=1)], dim=2) 
		hidden = torch.cat([hidden[:, 0, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1), 
			hidden[:, 1, :, :].view(self.num_layers, 1, batch_size, self.enc_hidden_dim).squeeze(dim=1)], dim=2) 
#		print("after cat hidden is {}".format(hidden.size()))
		hidden = hidden.view(self.num_layers, batch_size, 2 * self.enc_hidden_dim)
#		print("after view hidden is {}".format(hidden.size()))

		return output, hidden, enc_input

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
#		context = enc_outputs[:, -1, :].unsqueeze(dim=1).transpose(0, 1) 
		context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
							 enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
		concat = torch.cat([embedded, context], 2).to(device)
		output, hidden = self.gru(concat, dec_hidden)
		output = self.softmax(self.out(output[0].to(device)))    
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

		attn_placeholder = Variable(torch.zeros(batch_size, self.targ_max_sentence_len, self.src_max_sentence_len))

		return dec_outputs, hypotheses.transpose(0,1), attn_placeholder 


# class Attention(nn.Module): # previously Attention_Test
	
# 	""" Implements the attention mechanism by Bahdanau et al. (2015) """
	
# 	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
# 		super(Attention, self).__init__() 
# 		self.dec_hidden_dim = dec_hidden_dim
# 		self.input_dim = 2 * enc_hidden_dim + self.dec_hidden_dim
# 		self.attn = nn.Linear(self.input_dim, self.dec_hidden_dim).to(device)
# 		self.v = nn.Parameter(torch.rand(self.dec_hidden_dim))
# 		self.num_layers = num_layers 
# 		nn.init.normal_(self.v, mean=0, std=1. / math.sqrt(self.dec_hidden_dim))

# 	def forward(self, encoder_outputs, last_dec_hidden): 
# 		time_steps = encoder_outputs.size()[1]
# 		encoder_outputs, last_dec_hidden = encoder_outputs.to(device), last_dec_hidden.to(device) # [B, T, H], [L, B, H]
# 		batch_size = encoder_outputs.size()[0]
# 		v_broadcast = self.v.repeat(batch_size, 1, 1).to(device) # [B, 1, H]
# 		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1) # [B, L, H] -> [B, 1, H] -> [B, H] (take last layer)
# 		hidden_broadcast = last_dec_hidden.repeat(1, time_steps, 1).to(device) # [B, T, H]
# 		concat = torch.cat([encoder_outputs, hidden_broadcast], dim=2).to(device) # [B, T, 2H]
# 		energies = torch.tanh(self.attn(concat)).transpose(1, 2) # [B, T, H] -> [B, H, T]
# 		energies = torch.bmm(v_broadcast, energies).squeeze(1) # [B, 1, H] * [B, H, T] -> [B, 1, T] -> [B, T]
# 		attn_weights = F.softmax(energies, dim=1) # [B, T]

# 		return attn_weights


class Attention(nn.Module): 
	
	""" Implements the attention mechanism by Bahdanau et al. (2015) """
	
	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
		super(Attention, self).__init__() 
		self.dec_hidden_dim = dec_hidden_dim
		self.input_dim = 2 * enc_hidden_dim + self.dec_hidden_dim
		self.attn = nn.Linear(self.input_dim, self.dec_hidden_dim).to(device)
		self.v = nn.Parameter(torch.rand(self.dec_hidden_dim))
		self.num_layers = num_layers 
		nn.init.normal_(self.v, mean=0, std=1. / math.sqrt(self.dec_hidden_dim))

	def forward(self, encoder_outputs, last_dec_hidden, src_idx): 
		time_steps = encoder_outputs.size()[1]
		encoder_outputs, last_dec_hidden = encoder_outputs.to(device), last_dec_hidden.to(device) # [B, T, H], [L, B, H]
		batch_size = encoder_outputs.size()[0]
		v_broadcast = self.v.repeat(batch_size, 1, 1).to(device) # [B, 1, H]
		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1) # [B, L, H] -> [B, 1, H] -> [B, H] (take last layer)
		hidden_broadcast = last_dec_hidden.repeat(1, time_steps, 1).to(device) # [B, T, H]
		concat = torch.cat([encoder_outputs, hidden_broadcast], dim=2).to(device) # [B, T, 2H]
		energies = torch.tanh(self.attn(concat)).transpose(1, 2) # [B, T, H] -> [B, H, T]
		energies = torch.bmm(v_broadcast, energies).squeeze(1) # [B, 1, H] * [B, H, T] -> [B, 1, T] -> [B, T]
		energies.data.masked_fill_(src_idx == RESERVED_TOKENS['<PAD>'], -float('inf'))
		attn_weights = F.softmax(energies, dim=1) # [B, T]

		return attn_weights


# class DecoderAttnRNN(nn.Module): # previously DecoderAttnRNN_Test

# 	""" Decoder with attention (Bahdanau) """ 
	
# 	def __init__(self, rnn_cell_type, dec_hidden_dim, enc_hidden_dim, num_layers, dec_dropout, targ_vocab_size, src_max_sentence_len, targ_max_sentence_len, pretrained_word2vec):
# 		super(DecoderAttnRNN, self).__init__()
# 		self.dec_embed_dim = 300
# 		self.dec_hidden_dim = dec_hidden_dim 
# 		self.enc_hidden_dim = enc_hidden_dim
# 		self.src_max_sentence_len = src_max_sentence_len
# 		self.targ_max_sentence_len = targ_max_sentence_len
# 		self.targ_vocab_size = targ_vocab_size
# 		self.num_layers = num_layers 
# 		self.rnn_cell_type = rnn_cell_type 
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.attn = Attention(self.enc_hidden_dim, self.dec_hidden_dim, 
# 			num_annotations = self.src_max_sentence_len, num_layers=self.num_layers).to(device)
# 		if self.rnn_cell_type == 'gru':
# 			self.rnn = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		elif self.rnn_cell_type == 'lstm': 
# 			self.rnn = nn.LSTM(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		# self.gru = nn.GRU(self.dec_embed_dim + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size).to(device)
# 		self.softmax = nn.LogSoftmax(dim=1).to(device)

# 	def forward(self, dec_input, dec_hidden, enc_outputs):
# 		dec_input, dec_hidden = dec_input.to(device), dec_hidden.to(device) # [B], [L, B, H] 
# #		print("dec_input size is {}. dec_hidden size is {}".format(dec_input.size(), dec_hidden.size()))
# 		enc_outputs = enc_outputs.to(device) # [B * T * H] 
# #		print("enc_outputs size is {}".format(enc_outputs.size()))
# 		batch_size = dec_input.size()[0]
# 		embedded = self.embedding(dec_input).view(1, batch_size, -1) # [1, B, H]
# #		print("embedded size is {}".format(embedded.size()))
# 		attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden).unsqueeze(1) # [B, 1, T]
# #		print("attn_weights size is {}".format(attn_weights.size()))
# #		print("after bmm, attn_weights becomes context with size {}".format(attn_weights.bmm(enc_outputs).size())) 
# 		context = attn_weights.bmm(enc_outputs).transpose(0, 1) # [B, 1, T] * [B, T, H] = [B, 1, H] -> [1, B, H]
# 		concat = torch.cat([embedded, context], 2).to(device) # [1, B, 2H] 
# #		print("Embedded {} Context {} Concat {} dec_hidden".format(embedded.size(), context.size(), concat.size(), dec_hidden.size()))
# 		if self.rnn_cell_type == 'gru':
# 			output, hidden = self.rnn(concat, dec_hidden) # [1, B, H], [2, B, H] 
# 		elif self.rnn_cell_type == 'lstm':
# 			output, (hidden, memory) = self.rnn(concat, (dec_hidden, dec_hidden))		
# #		output, hidden = self.gru(concat, dec_hidden) # [1, B, H], [2, B, H] 
# 		output = self.softmax(self.out(output[0].to(device))) # [B, H] -> [B, V] 

# 		return output, hidden, attn_weights 


class DecoderAttnRNN(nn.Module): # previously DecoderAttnRNN_Test

	""" Decoder with attention (Bahdanau) """ 
	
	def __init__(self, rnn_cell_type, dec_hidden_dim, enc_hidden_dim, num_layers, dec_dropout, targ_vocab_size, src_max_sentence_len, targ_max_sentence_len, pretrained_word2vec):
		super(DecoderAttnRNN, self).__init__()
		self.dec_embed_dim = 300
		self.dec_hidden_dim = dec_hidden_dim 
		self.enc_hidden_dim = enc_hidden_dim
		self.src_max_sentence_len = src_max_sentence_len
		self.targ_max_sentence_len = targ_max_sentence_len
		self.targ_vocab_size = targ_vocab_size
		self.num_layers = num_layers 
		self.rnn_cell_type = rnn_cell_type 
		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
		self.attn = Attention(self.enc_hidden_dim, self.dec_hidden_dim, 
			num_annotations = self.src_max_sentence_len, num_layers=self.num_layers).to(device)
		if self.rnn_cell_type == 'gru':
			self.rnn = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
		elif self.rnn_cell_type == 'lstm': 
			self.rnn = nn.LSTM(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
		# self.gru = nn.GRU(self.dec_embed_dim + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size).to(device)
		self.softmax = nn.LogSoftmax(dim=1).to(device)

	def forward(self, dec_input, dec_hidden, enc_outputs, src_idx):
		dec_input, dec_hidden = dec_input.to(device), dec_hidden.to(device) # [B], [L, B, H] 
#		print("dec_input size is {}. dec_hidden size is {}".format(dec_input.size(), dec_hidden.size()))
		enc_outputs = enc_outputs.to(device) # [B * T * H] 
#		print("enc_outputs size is {}".format(enc_outputs.size()))
		batch_size = dec_input.size()[0]
		embedded = self.embedding(dec_input).view(1, batch_size, -1) # [1, B, H]
#		print("embedded size is {}".format(embedded.size()))
		attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden, src_idx=src_idx).unsqueeze(1) # [B, 1, T]
#		print("attn_weights size is {}".format(attn_weights.size()))
#		print("after bmm, attn_weights becomes context with size {}".format(attn_weights.bmm(enc_outputs).size())) 
		context = attn_weights.bmm(enc_outputs).transpose(0, 1) # [B, 1, T] * [B, T, H] = [B, 1, H] -> [1, B, H]
		concat = torch.cat([embedded, context], 2).to(device) # [1, B, 2H] 
#		print("Embedded {} Context {} Concat {} dec_hidden".format(embedded.size(), context.size(), concat.size(), dec_hidden.size()))
		if self.rnn_cell_type == 'gru':
			output, hidden = self.rnn(concat, dec_hidden) # [1, B, H], [2, B, H] 
		elif self.rnn_cell_type == 'lstm':
			output, (hidden, memory) = self.rnn(concat, (dec_hidden, dec_hidden))		
#		output, hidden = self.gru(concat, dec_hidden) # [1, B, H], [2, B, H] 
		output = self.softmax(self.out(output[0].to(device))) # [B, H] -> [B, V] 

		return output, hidden, attn_weights 


# class EncoderDecoderAttn(nn.Module): # previously EncoderDecoderAttention

# 	""" Encoder Decoder with Attention """

# 	def __init__(self, encoder, decoder, decoder_token2id): 
# 		super(EncoderDecoderAttn, self).__init__() 
# 		self.encoder = encoder 
# 		self.decoder = decoder 
# 		self.targ_vocab_size = self.decoder.targ_vocab_size
# 		self.src_max_sentence_len = self.encoder.src_max_sentence_len 
# 		self.targ_max_sentence_len = self.decoder.targ_max_sentence_len 

# 	def forward(self, src_idx, targ_idx, src_lens, targ_lens, teacher_forcing_ratio): 
		
# 		src_idx, targ_idx = src_idx.to(device), targ_idx.to(device) 
# 		src_lens, targ_lens = src_lens.to(device), targ_lens.to(device)
# 		batch_size = src_idx.size()[0]
# 		enc_outputs, enc_hidden = self.encoder(src_idx, src_lens)
# 		dec_hidden = enc_hidden 
# 		dec_outputs = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_vocab_size))
# 		hypotheses = Variable(torch.zeros(self.targ_max_sentence_len, batch_size))
# 		attn_weights_all = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_max_sentence_len))
# 		dec_output = targ_idx[:, 0] 

# 		for di in range(1, self.targ_max_sentence_len): 
# 			dec_output, dec_hidden, attn_weights = self.decoder(dec_output, dec_hidden, enc_outputs)
# 			dec_outputs[di] = dec_output 
# 			teacher_labels = targ_idx[:, di-1] 
# 			greedy_labels = dec_output.data.max(1)[1]
# 			dec_output = teacher_labels if random.random() < teacher_forcing_ratio else greedy_labels 
# 			hypotheses[di] = greedy_labels
# 			attn_weights_all[di] = attn_weights.squeeze(1)

# 		return dec_outputs, hypotheses.transpose(0,1), attn_weights_all.transpose(0,1)


class EncoderDecoderAttn(nn.Module): # previously EncoderDecoderAttention

	""" Encoder Decoder with Attention """

	def __init__(self, encoder, decoder, decoder_token2id): 
		super(EncoderDecoderAttn, self).__init__() 
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
		attn_weights_all = Variable(torch.zeros(self.targ_max_sentence_len, batch_size, self.targ_max_sentence_len))
		dec_output = targ_idx[:, 0] 

		for di in range(1, self.targ_max_sentence_len): 
#			dec_output, dec_hidden, attn_weights = self.decoder(dec_output, dec_hidden, enc_outputs)
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
		self.conv1_b = nn.Conv1d(300, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.conv2_b = nn.Conv1d(enc_hidden_dim, enc_hidden_dim, kernel_size=3, padding=1).to(device)
		self.dropout_val = dropout
		self.src_max_sentence_len = src_max_sentence_len
 

	def forward(self, enc_input, enc_input_lens):
		enc_input = enc_input.to(device)
		enc_input_lens = enc_input_lens.to(device)
		batch_size = enc_input.size()[0]
		embedded = self.embedding(enc_input)
		embedded = F.dropout(embedded, self.dropout_val)
		
		# 1st net
		hidden_1_a = self.conv1_a(embedded.transpose(1,2)).transpose(1,2)
		#print(hidden_1_a.shape)
		hidden_1_a.contiguous().view(-1, hidden_1_a.size(-1))
		hidden_1_a = F.leaky_relu(hidden_1_a.contiguous().view(-1, self.enc_embed_dim)
							   ).view(batch_size, -1, hidden_1_a.size(-1))
		hidden_2_a = self.conv2_a(hidden_1_a.transpose(1,2)).transpose(1,2)
		hidden_2_a = F.leaky_relu(hidden_2_a.contiguous().view(-1, hidden_2_a.size(-1))).view(
													batch_size, -1, hidden_2_a.size(-1))
		# 2nd net
		hidden_1_b = self.conv1_a(embedded.transpose(1,2)).transpose(1,2)
		hidden_1_b.contiguous().view(-1, hidden_1_b.size(-1))
		hidden_1_b = F.leaky_relu(hidden_1_b.contiguous().view(-1, self.enc_embed_dim)
							   ).view(batch_size, -1, hidden_1_b.size(-1))
		hidden_2_b = self.conv2_a(hidden_1_b.transpose(1,2)).transpose(1,2)
		hidden_2_b = F.leaky_relu(hidden_2_b.contiguous().view(-1, hidden_2_b.size(-1))).view(
													batch_size, -1, hidden_2_b.size(-1))
		hidden_2_b = hidden_2_b.view(-1, 2, batch_size, self.enc_hidden_dim)
		hidden_2_b = hidden_2_b.transpose(0,1)

		hidden_2_b = hidden_2_b[:, 0, :, :].squeeze(dim=1) + hidden_2_b[:, 1, :, :].squeeze(dim=1)
		
		return hidden_2_a , hidden_2_b.view(2,batch_size, -1)
	





    



### OLD CODE ### 

# class EncoderRNN(nn.Module):

# 	""" Vanilla RNN encoder, returns twice the original hidden dimension due to bidirectional 

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 

# 	""" 

# 	def __init__(self, enc_hidden_dim, num_layers, src_max_sentence_len, pretrained_word2vec):
# 		super(EncoderRNN, self).__init__()
# 		self.enc_embed_dim = 300
# 		self.enc_hidden_dim = enc_hidden_dim 
# 		self.src_max_sentence_len = src_max_sentence_len
# 		self.num_layers = num_layers
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.gru = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
# 						  batch_first=True, bidirectional=True).to(device)
	
# 	def forward(self, enc_input, enc_input_lens):
# 		enc_input = enc_input.to(device)
# 		enc_input_lens = enc_input_lens.to(device)
# 		batch_size = enc_input.size()[0]
# 		_, idx_sort = torch.sort(enc_input_lens, dim=0, descending=True)
# 		_, idx_unsort = torch.sort(idx_sort, dim=0)
# 		enc_input, enc_input_lens = enc_input.index_select(0, idx_sort), enc_input_lens.index_select(0, idx_sort)
# 		embedded = self.embedding(enc_input)
# 		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, enc_input_lens, batch_first=True)
# 		hidden = self.initHidden(batch_size).to(device)
# 		output, hidden = self.gru(embedded, hidden)
# 		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
# 														   total_length=self.src_max_sentence_len,
# 														   padding_value=RESERVED_TOKENS['<PAD>'])
# 		output = output.index_select(0, idx_unsort)
# 		hidden = hidden.index_select(1, idx_unsort).transpose(0, 1).contiguous().view(self.num_layers, batch_size, -1)

# 		return output, hidden

# 	def initHidden(self, batch_size):
# 		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)


# class EncoderSimpleRNN(nn.Module):

# 	""" Vanilla RNN encoder. Sums the bidirectional hidden/output instead of returning twice the hidden dimension """ 
	
# 	def __init__(self, enc_hidden_dim, num_layers, src_max_sentence_len, pretrained_word2vec):
# 		super(EncoderSimpleRNN, self).__init__()
# 		self.enc_embed_dim = 300
# 		self.enc_hidden_dim = enc_hidden_dim 
# 		self.src_max_sentence_len = src_max_sentence_len
# 		self.num_layers = num_layers
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.gru = nn.GRU(input_size=self.enc_embed_dim, hidden_size=self.enc_hidden_dim, num_layers=self.num_layers, 
# 						  batch_first=True, bidirectional=True).to(device)
	
# 	def forward(self, enc_input, enc_input_lens):
# 		enc_input = enc_input.to(device)
# 		enc_input_lens = enc_input_lens.to(device)
# 		batch_size = enc_input.size()[0]
# 		_, idx_sort = torch.sort(enc_input_lens, dim=0, descending=True)
# 		_, idx_unsort = torch.sort(idx_sort, dim=0)
# 		enc_input, enc_input_lens = enc_input.index_select(0, idx_sort), enc_input_lens.index_select(0, idx_sort)
# 		embedded = self.embedding(enc_input)
# 		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, enc_input_lens, batch_first=True)
# 		hidden = self.initHidden(batch_size).to(device)
# 		output, hidden = self.gru(embedded, hidden)
# 		output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, 
# 														   total_length=self.src_max_sentence_len,
# 														   padding_value=RESERVED_TOKENS['<PAD>'])
# 		output = output.index_select(0, idx_unsort)
# 		hidden = hidden.index_select(1, idx_unsort)
# 		output = output[:, :, :self.enc_hidden_dim] + output[:, :, self.enc_hidden_dim:]
# 		hidden = hidden.view(self.num_layers, 2, batch_size, self.enc_hidden_dim)
# 		hidden = hidden[:, 0, :, :].squeeze(dim=1) + hidden[:, 1, :, :].squeeze(dim=1)
# 		hidden = hidden.view(self.num_layers, batch_size, self.enc_hidden_dim)

# 		return output, hidden

# 	def initHidden(self, batch_size):
# 		return torch.zeros(2*self.num_layers, batch_size, self.enc_hidden_dim).to(device)


# class DecoderSimpleRNN(nn.Module):

# 	""" Vanilla decoder without attention, and final encoder hidden layer NOT passed to every time step of decoder 

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 
# 	""" 

# 	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
# 		super(DecoderSimpleRNN, self).__init__()
# 		self.dec_embed_dim = 300
# 		self.dec_hidden_dim = dec_hidden_dim 
# 		self.enc_hidden_dim = enc_hidden_dim
# 		self.targ_vocab_size = targ_vocab_size
# 		self.targ_max_sentence_len = targ_max_sentence_len
# 		self.num_layers = num_layers
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.gru = nn.GRU(self.dec_embed_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
# 		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size).to(device)
# 		self.softmax = nn.LogSoftmax(dim=1).to(device)

# 	def forward(self, dec_input, dec_hidden, enc_outputs): 
# 		dec_input = dec_input.to(device)
# 		dec_hidden = dec_hidden.to(device)
# 		enc_outputs = enc_outputs.to(device)
# 		batch_size = dec_input.size()[0]
# 		embedded = self.embedding(dec_input).view(1, batch_size, -1)
# 		dec_hidden = dec_hidden.view(self.num_layers, batch_size, self.dec_hidden_dim)
# 		output, hidden = self.gru(embedded, dec_hidden)
# 		output = self.softmax(self.out(output[0].to(device)))    

# 		return output, hidden


# class DecoderRNN(nn.Module):

# 	""" Vanilla decoder without attention, but final layer from encoder is repeatedly passed as input to each time step. 
# 		Handles output from EncoderRNN that returns twice the encoder hidden dimension.  

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 
# 	""" 

# 	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, targ_max_sentence_len, pretrained_word2vec):
# 		super(DecoderRNN, self).__init__()
# 		self.dec_embed_dim = 300
# 		self.dec_hidden_dim = dec_hidden_dim 
# 		self.enc_hidden_dim = enc_hidden_dim
# 		self.targ_vocab_size = targ_vocab_size
# 		self.targ_max_sentence_len = targ_max_sentence_len
# 		self.num_layers = num_layers
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.gru = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
# 		self.out = nn.Linear(dec_hidden_dim, self.targ_vocab_size).to(device)
# 		self.softmax = nn.LogSoftmax(dim=1).to(device)

# 	def forward(self, dec_input, dec_hidden, enc_outputs): 
# 		dec_input = dec_input.to(device)
# 		dec_hidden = dec_hidden.to(device)
# 		enc_outputs = enc_outputs.to(device)
# 		batch_size = dec_input.size()[0]
# 		embedded = self.embedding(dec_input).view(1, batch_size, -1)
# 		context = torch.cat([enc_outputs[:, -1, :self.enc_hidden_dim], 
# 							 enc_outputs[:, 0, self.enc_hidden_dim:]], dim=1).unsqueeze(0)
# 		concat = torch.cat([embedded, context], 2).to(device)
# 		output, hidden = self.gru(concat, dec_hidden)
# 		output = self.softmax(self.out(output[0].to(device)))    
# 		return output, hidden


# class Attention(nn.Module): 
	
# 	""" Implements the attention mechanism by Bahdanau et al. (2015) 

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 
# 	"""
	
# 	def __init__(self, enc_hidden_dim, dec_hidden_dim, num_annotations, num_layers): 
# 		super(Attention, self).__init__() 
# #		self.num_annotations = num_annotations
# 		self.dec_hidden_dim = dec_hidden_dim
# 		self.input_dim = enc_hidden_dim + self.dec_hidden_dim
# 		self.attn = nn.Linear(self.input_dim, self.dec_hidden_dim).to(device)
# 		self.v = nn.Parameter(torch.rand(self.dec_hidden_dim))
# 		self.num_layers = num_layers 
# 		nn.init.normal_(self.v, mean=0, std=1. / math.sqrt(self.dec_hidden_dim))

# 	def forward(self, encoder_outputs, last_dec_hidden): 
# 		# print("Attention module receives encoder outputs of size {} and last_dec_hidden of size {}".format(
# 		# 	encoder_outputs.size(), last_dec_hidden.size()))
# 		time_steps = encoder_outputs.size()[1]
# 		encoder_outputs, last_dec_hidden = encoder_outputs.to(device), last_dec_hidden.to(device) # [B, T, H], [L, B, H]
# 		batch_size = encoder_outputs.size()[0]
# 		v_broadcast = self.v.repeat(batch_size, 1, 1).to(device) # [B, 1, H]
# 		last_dec_hidden = last_dec_hidden.transpose(0, 1)[:, -1, :].unsqueeze(1) # [B, L, H] -> [B, 1, H] -> [B, H] (take last layer)
# 		hidden_broadcast = last_dec_hidden.repeat(1, time_steps, 1).to(device) # [B, T, H]
# 		concat = torch.cat([encoder_outputs, hidden_broadcast], dim=2).to(device) # [B, T, 2H]
# 		energies = torch.tanh(self.attn(concat)).transpose(1, 2) # [B, T, H] -> [B, H, T]
# 		energies = torch.bmm(v_broadcast, energies).squeeze(1) # [B, 1, H] * [B, H, T] -> [B, 1, T] -> [B, T]
# 		attn_weights = F.softmax(energies, dim=1) # [B, T]

# 		return attn_weights


# class DecoderAttnRNN(nn.Module):

# 	""" Decoder with attention (Bahdanau) 

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 
# 	""" 
	
# 	def __init__(self, dec_hidden_dim, enc_hidden_dim, num_layers, targ_vocab_size, src_max_sentence_len, targ_max_sentence_len, pretrained_word2vec):
# 		super(DecoderAttnRNN, self).__init__()
# 		self.dec_embed_dim = 300
# 		self.dec_hidden_dim = dec_hidden_dim 
# 		self.enc_hidden_dim = enc_hidden_dim
# 		self.src_max_sentence_len = src_max_sentence_len
# 		self.targ_max_sentence_len = targ_max_sentence_len
# 		self.targ_vocab_size = targ_vocab_size
# 		self.num_layers = num_layers 
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		self.attn = Attention(self.enc_hidden_dim, self.dec_hidden_dim, 
# 							  num_annotations = self.src_max_sentence_len, num_layers=self.num_layers).to(device)
# 		self.gru = nn.GRU(self.dec_embed_dim + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers).to(device)
# 		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size).to(device)
# 		self.softmax = nn.LogSoftmax(dim=1).to(device)

# 	def forward(self, dec_input, dec_hidden, enc_outputs):
# 		dec_input, dec_hidden = dec_input.to(device), dec_hidden.to(device) # [B], [L, B, H] 
# #		print("dec_input size is {}. dec_hidden size is {}".format(dec_input.size(), dec_hidden.size()))
# 		enc_outputs = enc_outputs.to(device) # [B * T * H] 
# #		print("enc_outputs size is {}".format(enc_outputs.size()))
# 		batch_size = dec_input.size()[0]
# 		embedded = self.embedding(dec_input).view(1, batch_size, -1) # [1, B, H]
# #		print("embedded size is {}".format(embedded.size()))
# 		attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden).unsqueeze(1) # [B, 1, T]
# #		print("attn_weights size is {}".format(attn_weights.size()))
# #		print("after bmm, attn_weights becomes context with size {}".format(attn_weights.bmm(enc_outputs).size())) 
# 		context = attn_weights.bmm(enc_outputs).transpose(0, 1) # [B, 1, T] * [B, T, H] = [B, 1, H] -> [1, B, H]
# 		concat = torch.cat([embedded, context], 2).to(device) # [1, B, 2H] 
# 		output, hidden = self.gru(concat, dec_hidden) # [1, B, H], [2, B, H] 
# 		output = self.softmax(self.out(output[0].to(device))) # [B, H] -> [B, V] 

# 		return output, hidden, attn_weights 

# class DecoderRNN_Test(nn.Module):

# 	""" Decoder with attention (Bahdanau) 

# 		*** TODO *** 
# 		- Haven't retested after major bug fix. Retry later. 
# 	""" 
	
# 	def __init__(self, rnn_cell_type, attn, dec_hidden_dim, enc_hidden_dim, num_layers, dec_dropout, targ_vocab_size, src_max_sentence_len, targ_max_sentence_len, pretrained_word2vec):
# 		super(DecoderRNN_Test, self).__init__()
# 		self.dec_embed_dim = 300
# 		self.dec_hidden_dim = dec_hidden_dim 
# 		self.enc_hidden_dim = enc_hidden_dim
# 		self.src_max_sentence_len = src_max_sentence_len
# 		self.targ_max_sentence_len = targ_max_sentence_len
# 		self.targ_vocab_size = targ_vocab_size
# 		self.num_layers = num_layers 
# 		self.rnn_cell_type = rnn_cell_type 
# 		self.attn = attn
# 		self.embedding = nn.Embedding.from_pretrained(pretrained_word2vec, freeze=True).to(device)
# 		if self.attn: 
# 			self.attn = Attention_Test(self.enc_hidden_dim, self.dec_hidden_dim, 
# 				num_annotations = self.src_max_sentence_len, num_layers=self.num_layers).to(device)
# 		if self.rnn_cell_type == 'gru':
# 			self.rnn = nn.GRU(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		elif self.rnn_cell_type == 'lstm': 
# 			self.rnn = nn.LSTM(self.dec_embed_dim + 2 * self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		# self.gru = nn.GRU(self.dec_embed_dim + self.enc_hidden_dim, self.dec_hidden_dim, num_layers=self.num_layers, dropout=dec_dropout).to(device)
# 		self.out = nn.Linear(self.dec_hidden_dim, self.targ_vocab_size).to(device)
# 		self.softmax = nn.LogSoftmax(dim=1).to(device)

# 	def forward(self, dec_input, dec_hidden, enc_outputs):
# 		dec_input, dec_hidden = dec_input.to(device), dec_hidden.to(device) # [B], [L, B, H] 
# #		print("dec_input size is {}. dec_hidden size is {}".format(dec_input.size(), dec_hidden.size()))
# 		enc_outputs = enc_outputs.to(device) # [B * T * H] 
# #		print("enc_outputs size is {}".format(enc_outputs.size()))
# 		batch_size = dec_input.size()[0]
# 		embedded = self.embedding(dec_input).view(1, batch_size, -1) # [1, B, H]
# #		print("embedded size is {}".format(embedded.size()))
# 		if self.attn: 
# 			attn_weights = self.attn(encoder_outputs=enc_outputs, last_dec_hidden=dec_hidden).unsqueeze(1) # [B, 1, T]
# 	#		print("attn_weights size is {}".format(attn_weights.size()))
# 	#		print("after bmm, attn_weights becomes context with size {}".format(attn_weights.bmm(enc_outputs).size())) 
# 			context = attn_weights.bmm(enc_outputs).transpose(0, 1) # [B, 1, T] * [B, T, H] = [B, 1, H] -> [1, B, H]
# 		else: 
# 			context = enc_outputs[:, -1, :].unsqueeze(dim=1).transpose(0, 1) 
# 		concat = torch.cat([embedded, context], 2).to(device) # [1, B, 2H] 
# #		print("Embedded {} Context {} Concat {} dec_hidden".format(embedded.size(), context.size(), concat.size(), dec_hidden.size()))
# 		if self.rnn_cell_type == 'gru':
# 			output, hidden = self.rnn(concat, dec_hidden) # [1, B, H], [2, B, H] 
# 		elif self.rnn_cell_type == 'lstm':
# 			output, (hidden, memory) = self.rnn(concat, (dec_hidden, dec_hidden))		
# #		output, hidden = self.gru(concat, dec_hidden) # [1, B, H], [2, B, H] 
# 		output = self.softmax(self.out(output[0].to(device))) # [B, H] -> [B, V] 

# 		if self.attn: 
# 			return output, hidden, attn_weights 
# 		else: 
# 			return output, hidden 

