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
from datetime import datetime
from utils import progress_bar
import os
from dataloader import *
from Model_2 import *
import csv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


test_x=np.load('test_new.npy', allow_pickle=True, encoding='bytes')#[1:5]
test_dataset=MyDataset_test(test_x)
test_dataloader = DataLoader(
    test_dataset, # The dataset
    batch_size=30,      # Batch size
    shuffle=False,      # Shuffles the dataset at every epoch
    collate_fn=my_collate_test,
    pin_memory=True,   # Copy data to CUDA pinned memo so that they can be transferred to the GPU very fast
    num_workers=0  )


def test(model,test_loader):
    model.eval()
    with torch.no_grad():
        with open("prediction.csv", 'w') as f:
            f_writer = csv.writer(f, delimiter=',')
            f_writer.writerow(['Id', 'Predicted'])
            id=0
            for (batch_num, collate_output) in enumerate(test_loader):
    #               with torch.autograd.set_detect_anomaly(True):
                X=collate_output
                speech_len=torch.LongTensor([len(i) for i in X])
                speech_input=pad_sequence(X,batch_first=False)
                speech_input = speech_input.to(device)
                predictions = model(speech_input, speech_len ,0.9,train=False)#torch.Size([N, L, V])
                # predictions = F.gumbel_softmax(predictions, dim=2)
                pred_indx = torch.argmax(predictions,dim=2)#torch.Size([N, L])
                pred_str_list=transform_index_to_letter_test(pred_indx)
                for i in pred_str_list:
                    row=[]
                    row.append(str(id))
                    row.append(i)
                    f_writer.writerow(row)
                    id += 1

        #         # net.eval()
        #         # with open("pre.csv",'w') as f:
        #         #     f_writer = csv.writer(f, delimiter=',')
        #         #     f_writer.writerow(['id', 'label'])
        #         #     for batch_idx, (inputs, id) in enumerate(dataloader_test):
        #         #         res = []
        #         #         res.append(id.item())
        #         #         inputs = inputs.to(device)
        #         #         outputs = net(inputs)
        #         #         _, predicted = outputs.max(1)
        #         #         res.append(predicted.item())
        #         #         f_writer.writerow(res)

print('==> Building model..')
model = Seq2Seq(input_dim=40,vocab_size=len(letter_list),encoder_hidden_dim=256,embeding=256,decoder_hidden_dim=512)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduce=None).to(device)

# print('==> Resuming from checkpoint..')
    # checkpoint = torch.load('./checkpoint/ckpt.pth')
    # model.load_state_dict(checkpoint['net'])
    # start_epoch = checkpoint['epoch']
# if os.path.isdir('checkpoint'):
print("resum model..")
PATH='/home/ubuntu//DL/hw4p2/checkpoint/All_p=0.7/epoch-5_loss-0.09049_dis-7.11392.pth'
# PATH='epoch_2-loss_0.6997552-dis_85.6242626.pth'
checkpoint = torch.load(PATH)#,map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['net'])
print("model loaded!")
print("start predicting.....")
test(model,test_dataloader)
print("finished predicting!")