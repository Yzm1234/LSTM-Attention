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
import pickle as pk
from torch.utils.data import DataLoader, Dataset
import time
import Levenshtein


letter_list = ['0','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


############ Transformer Functions #####################
def transform_letter_to_index(transcript, letter_list):
    train_Y=[]
    for sen in transcript:
        res=[letter_list.index('<sos>')]
        for word in sen:
            for letter in word.decode('utf-8'):
                res.append(letter_list.index(letter))
            res.append(letter_list.index(' '))
        res.pop()
        res.append(letter_list.index('<eos>'))
        train_Y.append(res)
    return train_Y


def transform_index_to_letter(pred_indx):#input (N,L)
    res=[]
    for row in pred_indx:
        string=''
        for i in row:
            if letter_list[i]=='<eos>':
                string += letter_list[i]
                break
            else:
                string+=letter_list[i]
        res.append(string)
    return res

def transform_index_to_letter_test(pred_indx):#input (N,L)
    res=[]
    for row in pred_indx:
        string=''
        for i in row:
            if letter_list[i]=='<eos>' or letter_list[i]=='0':
                break
            else:
                string+=letter_list[i]
        res.append(string)
    return res

def transform_index_to_letter_Train(pred_indx):#input (N,L)
    res=[]
    for row in pred_indx:
        string=''.join(letter_list[i] for i in row)
        res.append(string)
    return res


############# DataLoader ##############################
class MyDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        assert len(x) == len(y)
        self._x = x
        self._y = y

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y

class MyDataset_test(Dataset):
    def __init__(self, x):
        super().__init__()
        self._x = x

    def __len__(self):
        return len(self._x)

    def __getitem__(self, index):
        x = self._x[index]
        return x

def my_collate(batch):
    X = [torch.tensor(x) for x,_ in batch]
    Y = [torch.LongTensor(y) for _,y in batch]
    return [X, Y]


def my_collate_test(batch):
    X = [torch.tensor(x) for x in batch]
    return X

