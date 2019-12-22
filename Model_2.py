import torch
from torch import nn
from torch.nn.utils.rnn import *
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn.functional as F
import random
import time
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from dataloader import *
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
import Levenshtein

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class pBLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.pblstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

    def forward(self, x, x_len):  # x (L x N x input_dim)
        '''
        :param x :(N,T) input to the pBLSTM # input is a padded sequence
        :param x_len :(N,) input to the pBLSTM
        :return output: (N,T,H) encoded sequence from pyramidal Bi-LSTM
        '''
        batch_size = x.shape[1]
        x_pack = utils.rnn.pack_padded_sequence(x, lengths=x_len, batch_first=False, enforce_sorted=False)
        outputs, _ = self.pblstm(x_pack)
        x_pad, x_len = utils.rnn.pad_packed_sequence(outputs)  # x_pad:(L,N,512)
        feature_dim = x_pad.shape[2]
        max_len = max(x_len)  # L

        if max_len % 2 != 0:
            x_pad = x_pad[0:-1]
            x_len[torch.argmax(x_len)] = max_len - 1
            max_len = max(x_len)

        x_len = [i // 2 for i in x_len]
        x_pad = torch.transpose(x_pad, 0, 1)  # (N,L,512)
        x_pad = x_pad.contiguous().view(batch_size, max_len // 2, 2 * feature_dim)  # (N,L/2,512*2)
        x_pad = torch.transpose(x_pad, 0, 1)  # x_pad:(L/2,N,512*2)
        return x_pad, x_len


class Encoder(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, value_size=128, key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=256, num_layers=1, bidirectional=True)
        # Here you need to define the blocks of pBLSTMs
        self.pblstm = pBLSTM(input_dim=256 * 2, hidden_dim=256)
        self.pblstm_2 = pBLSTM(input_dim=512 * 2, hidden_dim=256)
        self.key_network = nn.Linear(hidden_dim * 4, value_size)
        self.value_network = nn.Linear(hidden_dim * 4, key_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        # Use the outputs and pass it through the pBLSTM blocks
        pad_out, pad_len = utils.rnn.pad_packed_sequence(outputs)  # L,N,2H

        ly1_out, ly1_len = self.pblstm(pad_out, pad_len)
        ly2_out, ly2_len = self.pblstm_2(ly1_out, ly1_len)
        ly3_out, ly3_len = self.pblstm_2(ly2_out, ly2_len)

        keys = self.key_network(ly3_out)  #:L x N x 128
        # print("x after pblstm",linear_input)
        value = self.value_network(ly3_out)  # L x N x 128
        # print("encoder key and value size", keys.size(), value.size())
        return keys, value, torch.LongTensor(ly3_len)

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
    def forward(self, query, key, value, lens):
        '''
        :param query :(N,context_size) Query is the output of LSTMCell from Decoder
        # query 是一个time step的vector所以 L是1
        :param key: (T,N,key_size) Key Projection from Encoder per time step
        :param value: (T,N,value_size) Value Projection from Encoder per time step
        :param lengs: (N,), lengths of source sequences
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted
        '''
        #Compute (N, T) attention logits. "bmm" stands for "batch matrix multiplication".
        # Input/output shape of bmm: (N, T, H), (N, H, 1) -> (N, T, 1)
        '''
            def forward(self, query, context, lengths):
        """
        :param query: (N, H), decoder state of a single timestep
        :param context: (N, T, H), encoded source sequences
        :param lengths: (N,), lengths of source sequences
        :returns: (N, H) attended source context, and (N, T) attention vectors
        """
        # Compute (N, T) attention logits. "bmm" stands for "batch matrix multiplication".
        # Input/output shape of bmm: (N, T, H), (N, H, 1) -> (N, T, 1)
        attention = torch.bmm(context, query.unsqueeze(2)).squeeze(2)
        # Create an (N, T) boolean mask for all padding positions
        # Make use of broadcasting: (1, T), (N, 1) -> (N, T)
        mask = torch.arange(context.size(1)).unsqueeze(0) >= lengths.unsqueeze(1)
        # Set attention logits at padding positions to negative infinity.
        attention.masked_fill_(mask, -1e9)
        # Take softmax over the "source length" dimension.
        attention = nn.functional.softmax(attention, dim=1)
        # Compute attention-weighted sum of context vectors
        # Input/output shape of bmm: (N, 1, T), (N, T, H) -> (N, 1, H)
        out = torch.bmm(attention.unsqueeze(1), context).squeeze(1)
        return out, attention'''
        key=torch.transpose(key,0,1)#(N,T,128)
        # print("key shape",key.shape)
        # print("query shape",query.shape)
        value=torch.transpose(value,0,1)#(N,T,128)
        energy = torch.bmm(key, query.unsqueeze(2)).squeeze(2).to(device)#(N,T)
        # print("key",key.shape)
        # print("lens")
        mask=(torch.arange(value.size(1)).unsqueeze(0)>=lens.unsqueeze(1)).to(device)#(N,T)
        # print("mask",mask)
        energy.masked_fill_(mask, -1e9).to(device)
        attention = F.softmax(energy,dim=1)#(N,T)
        context = torch.bmm(attention.unsqueeze(1),value).squeeze(1)#(N,128)
        return context, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embeding, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Decoder, self).__init__()
        self.hidden_dim=hidden_dim
        self.embedding = nn.Embedding(vocab_size, embeding)
        self.linear_query=nn.Linear(hidden_dim, key_size)
        self.lstm1 = nn.LSTMCell(input_size=embeding + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.isAttended = isAttended
        if (isAttended):
            self.attention = Attention()
        self.character_prob = nn.Linear(hidden_dim + value_size, vocab_size)

    def forward(self, key, values, speech_len,p, text=None, train=True):
        #     '''
        #     :param key :(T,N,key_size) Output of the Encoder Key projection layer
        #     :param values: (T,N,value_size) Output of the Encoder Value projection layer
        #     :param text: (N,text_len) Batch input of text with text_length # ground truth labels of all time steps
        #     :param train: Train or eval mode
        #     :return predictions: Returns the character perdiction probability
        #     '''
        #         print("Decoder forward",text.shape)
        batch_size = key.shape[1]
        if (train):
            max_len = text.shape[1]
            embeddings = self.embedding(text)  # (N，L ,H)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]  # 每层lstm一个 hidden state
        prediction = torch.ones(batch_size,1).to(device)*letter_list.index('<sos>')##??????????????????
        output = torch.zeros(batch_size,self.hidden_dim).to(device)

        #         print(":::::::value_len::::::::::",len(values))
        for i in range(max_len):  # loop through time steps in decoder
            #               '''
            #               Here you should implement Gumble noise and teacher forcing techniques
            #               '''
            if (train):
                num = random.uniform(0, 1)
                if num > p:
                    char_embed = embeddings[:, i, :]
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))# the last dimension

            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            # When attention is True you should replace the values[i,:,:] with the context you get from attention

            query=self.linear_query(output)
            context, attetion = self.attention(query, key, values, speech_len)  # context of shape(N,128)
            inp = torch.cat([char_embed, context], dim=1)  # (N,128*2)

            hidden_states[0] = self.lstm1(inp, hidden_states[0]) # hidden_states[0]: a tuple (h,c)

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0] # hidden_states[0][1]: h -> query
            prediction = self.character_prob(torch.cat([output, context], dim=1))

            predictions.append(prediction.unsqueeze(1))
            # predictions are logist of (N,len(voc))
            # print("predictions shape after decoder",torch.cat(predictions, dim=1).shape)
        return torch.cat(predictions, dim=1)# 4x68

class Seq2Seq(nn.Module):
    def __init__(self,input_dim,vocab_size,encoder_hidden_dim,embeding,decoder_hidden_dim,value_size=128, key_size=128,isAttended=True):
        super(Seq2Seq,self).__init__()

        self.encoder = Encoder(input_dim, encoder_hidden_dim)
        self.decoder = Decoder(vocab_size, embeding, decoder_hidden_dim)
    def forward(self,speech_input, speech_len, p,text_input=None,train=True):
        # speech_input: (L,N,H)
        # speech_len: (N,)
#         print("Seq2Seq",speech_input.shape, len(speech_len))
        key, value,shorten_len = self.encoder(speech_input, speech_len)
        if(train):
            predictions = self.decoder(key, value,shorten_len, p,text=text_input)
        else:
            predictions = self.decoder(key, value,shorten_len, p,text=torch.LongTensor([letter_list.index('<sos>')]), train=False)

        return predictions
