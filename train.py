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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

######### Loading dataset ##############################
print("Loading dataset.....")
train_x=np.load('train_new.npy', allow_pickle=True,encoding='bytes')#[1:3]
train_y=np.load('train_transcripts.npy', allow_pickle=True, encoding='bytes')#[1:3]

val_x=np.load('dev_new.npy', allow_pickle=True, encoding='bytes')#[1:30]
val_y=np.load('dev_transcripts.npy', allow_pickle=True, encoding='bytes')#[1:30]

train_Y=transform_letter_to_index(train_y, letter_list)
val_Y=transform_letter_to_index(val_y, letter_list)
train_dataset=MyDataset(train_x,train_Y)
val_dataset=MyDataset(val_x,val_Y)

train_dataloader = DataLoader(
    train_dataset, # The dataset
    batch_size=50,      # Batch size
    shuffle=True,      # Shuffles the dataset at every epoch
    collate_fn=my_collate,
    pin_memory=True,   # Copy data to CUDA pinned memo so that they can be transferred to the GPU very fast
    num_workers=0  )
val_dataloader = DataLoader(
    val_dataset, # The dataset
    batch_size=80,      # Batch size
    shuffle=False,      # Shuffles the dataset at every epoch
    collate_fn=my_collate,
    pin_memory=True,   # Copy data to CUDA pinned memo so that they can be transferred to the GPU very fast
    num_workers=0  )

print("Finished Loading!")

######## Building Model ################################
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
PATH='/home/ubuntu/DL/hw4p2/checkpoint/p=0.6/epoch-19_loss-0.05549_dis-13.06962.pth'
# PATH='epoch_2-loss_0.6997552-dis_85.6242626.pth'# CPU path
checkpoint = torch.load(PATH)#,map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['net'])
print("model loaded!")
def train(model, train_loader, num_epochs, criterion, optimizer,p):
    print("epoch:", num_epochs)
    loss_sum = 0
    model.train()
    for (batch_num, collate_output) in enumerate(train_loader):
        with torch.autograd.set_detect_anomaly(True):

            #                 speech_input, text_input, speech_len, text_len = collate_output
            X, Y = collate_output
            speech_len = torch.LongTensor([len(i) for i in X]).to(device)
            text_len = torch.LongTensor([len(i) for i in Y]).to(device)
            speech_input = pad_sequence(X, batch_first=False)
            text_input = pad_sequence(Y, batch_first=True)
            speech_input = speech_input.to(device)
            text_input = text_input.to(device)
            predictions = model(speech_input, speech_len,p, text_input, train=True)  # torch.Size([N, L, V])
            # 对齐label和 pred
            text_input = text_input[:, 1::]
            predictions=predictions[:,0:-1,:]
            if batch_num % 100 == 1:
                pred_indx = torch.argmax(predictions, dim=2)
                pred_str_list = transform_index_to_letter(pred_indx)
                print("pred_str_list", pred_str_list)

            #predictions =F.gumbel_softmax(predictions, dim=2)

            #true_label_strlist = transform_index_to_letter_Train(text_input)
            mask = torch.zeros(text_input.size()).to(device)
            #                 print("len(text_len)",len(text_len))
            for i in range(len(text_len)):
                mask[i, :text_len[i]-1] = 1

            mask = mask.view(-1).to(device)
            #predictions Nx(LxV)


            predictions = predictions.contiguous().view(-1, predictions.size(-1))  # torch.Size([N*L, V])
            #                 text_input = text_input.T
            # print("text_input shape before flaten",text_input.shape)
            text_input = text_input.contiguous().view(-1)

            # print("text_input after flaten:",text_input.shape)
            loss = criterion(predictions, text_input)
            masked_loss = torch.sum(loss * mask)
            masked_loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()

            current_loss = float(masked_loss.item()) / int(torch.sum(mask).item())
            loss_sum+=current_loss


            print("%d Train" % num_epochs)
            progress_bar(batch_num, len(train_dataloader), 'Loss: %.7f ' % (loss_sum / (batch_num + 1)))
    return loss_sum / (batch_num + 1)



def val(model,val_loader, num_epochs, criterion,p,path):
    print("epoch:",num_epochs)
    dis_sum = 0
    len_sum=0
    model.eval()
    with torch.no_grad():
        for (batch_num, collate_output) in enumerate(val_loader):
            X,Y=collate_output
            speech_len=torch.LongTensor([len(i) for i in X])
            text_len=torch.LongTensor([len(i) for i in Y])
            print(len(X))
            len_sum+=len(X)
            speech_input=pad_sequence(X,batch_first=False)
            text_input=pad_sequence(Y,batch_first=True)
            speech_input = speech_input.to(device)
            text_input = text_input.to(device)
            predictions = model(speech_input, speech_len,p=0,train=False)[:,0:-1,:]#torch.Size([N, L, V])
            #predictions_for_loss = model(speech_input, speech_len,p,text_input,train=True)[:,0:-1,:]#torch.Size([N, L, V])
            text_input=text_input[:, 1::]
            ##  Levenshtein Distance
            #predictions_for_loss=F.gumbel_softmax(predictions_for_loss, dim=2)
            # predictions = F.gumbel_softmax(predictions, dim=2)
            pred_indx = torch.argmax(predictions, dim=2)  # torch.Size([N, L])
            pred_str_list = transform_index_to_letter(pred_indx)
            # print("pred_str_list",pred_str_list)
            true_str_list = transform_index_to_letter(text_input)
            batch_dis=0
            for i in range(len(pred_str_list)):
                dis=Levenshtein.distance(pred_str_list[i], true_str_list[i])
                batch_dis+=dis
                dis_sum+=dis
            ave_dis=batch_dis/len(X)
            epoch_ave=dis_sum/len_sum

            ## Crossentropy Loss
            # mask = torch.zeros(text_input.size()).to(device)
            # for i in range(len(text_len)):
            #     mask[i,:text_len[i]-1] = 1
            # mask = mask.view(-1).to(device)
            # predictions_for_loss = predictions_for_loss.contiguous().view(-1, predictions_for_loss.size(-1))
            # text_input = text_input.contiguous().view(-1)
            # loss = criterion(predictions_for_loss, text_input)
            # masked_loss = torch.sum(loss*mask)
            # current_loss = float(masked_loss.item())/int(torch.sum(mask).item())
            # loss_sum+=current_loss
            ## print
            if batch_num % 200 == 1:
                for i in range(len(pred_str_list)):
                    print("pred_str_list", pred_str_list[i])
                    print("true_str_list",true_str_list[i],'\n')
            print("%d Val" % num_epochs)
            progress_bar(batch_num, len(val_loader), ' Batch Dis: %.3f | Epoch Dis: %.3f  '% (ave_dis,epoch_ave))


    # Save checkpoint.
    dis = epoch_ave
    print("Saving....")
    state = {
        'net': model.state_dict(),
        'dis': dis,
        'epoch': num_epochs,
    }

    file_name=('epoch_%d-dis_%.7f.pth' % (num_epochs,dis))
    torch.save(state, path+file_name)
    return dis

# test_dataset=MyDataset_test(val_x)
# test_dataloader = DataLoader(
#     test_dataset, # The dataset
#     batch_size=30,      # Batch size
#     shuffle=False,      # Shuffles the dataset at every epoch
#     collate_fn=my_collate_test,
#     pin_memory=True,   # Copy data to CUDA pinned memo so that they can be transferred to the GPU very fast
#     num_workers=0)

import csv
# def val(model,test_loader,val_y):
#     model.eval()
#     pred_list=[]
#     with torch.no_grad():
#         for (batch_num, collate_output) in enumerate(test_loader):
# #               with torch.autograd.set_detect_anomaly(True):
#             X=collate_output
#             speech_len=torch.LongTensor([len(i) for i in X])
#             speech_input=pad_sequence(X,batch_first=False)
#             speech_input = speech_input.to(device)
#             predictions = model(speech_input, speech_len ,0.9,train=False)#torch.Size([N, L, V])
#             # predictions = F.gumbel_softmax(predictions, dim=2)
#             pred_indx = torch.argmax(predictions,dim=2)#torch.Size([N, L])
#             pred_str_list=transform_index_to_letter_test(pred_indx)
#             for i in pred_str_list:
#                 pred_list.append(i)
#
#     true_list = []
#     for s in val_y:
#         string = ''.join(i.decode('utf-8') + ' ' for i in s)[:-1]
#         true_list.append(string)
#
#     dis = 0
#     for i in range(len(true_list)):
#         dis += Levenshtein.distance(pred_list[i], true_list[i])
#
#     ave_dis = dis / len(true_list)
#     # print("average distance is:", ave_dis)
#     return ave_dis


# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
# now = datetime.now().strftime("%H:%M:%S")
now="All_p=0.7"
os.mkdir('checkpoint/'+now)
path='./checkpoint/'+now+'/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
p=0.7
for i in range(20):
    print("train")
    train(model, train_dataloader, i, criterion, optimizer,p)
    print("val")
    val(model, val_dataloader, i, criterion, p, path)
