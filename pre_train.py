"""
我有预训练的策略，这个策略好不好另说。
我当时有跟"Scientific Data"沟通过，想让他直接帮我看代码，但是他没看我想让他看的代码。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import math


torch.manual_seed(66)

class Swish(nn.Module):
    def __init__(self,beta=1.0):
        super().__init__()
        self.beta = beta
    def forward(self,x):
        return x*torch.sigmoid(self.beta*x)

class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.word_embed = nn.Embedding(1552,512)
        self.char_embed = nn.Embedding(80,512)
        self.char_proj = nn.Linear(512,1)
        self.word_proj=nn.Linear(512,1)
        self.swish=Swish()
        self.sigmoid=torch.nn.Sigmoid()
        self.classiffier=nn.Linear(512,1552)
        
    def forward(self,x):
        word_embed=self.word_embed(x[0])
        char_weights_list=[]
        for item in range(len(x[1])):
            # word_char_embed=self.char_embed(x[1][item].long())
            word_char_embed=self.char_embed(x[1][item].long())
            pre_attn=self.char_proj(word_char_embed)
            pre_attn=pre_attn.transpose(1,0)
            attn_weights=pre_attn@word_char_embed
            # print("word embed shape:",word_embed.shape)
            # print("word 1 char embed shape:",word_char_embed.shape)
            # print("pre attn:",pre_attn.shape)
            # print("attn weights shape:",attn_weights.shape)
            char_weights_list.append(attn_weights)
        # print(len(char_weights_list))
        char_embed=torch.concat(char_weights_list,dim=0)
        all_embed=self.sigmoid(word_embed*char_embed)
        word_attn=self.swish(self.word_proj(all_embed))
        word_attn=word_attn.transpose(1,0)
        word_weights=word_attn@all_embed
        res=self.classiffier(word_weights)
        return all_embed,res


if __name__=='__main__':
    # biosses corpus
    with open("corpus",'r') as f:
        data = f.readlines()
    word2id={}
    id2word={}
    char2id={}
    id2char={}
    for item in data[0].split(' '):
        pre_len=len(word2id)
        if item not in word2id.keys():
            word2id[item]=pre_len
            id2word[pre_len]=item
        for i in range(len(item)):
            pre_l=len(char2id)
            if item[i] not in char2id.keys():
                char2id[item[i]]=pre_l
                id2char[pre_l]=item[i]
    print("word2id length:",len(word2id))
    # print(id2word)
    print("char2id length:",len(char2id))
    # print(id2char)
    i=0
    j=9
    mid=math.floor((i+j)/2)
    # print(mid)
    train=[]
    print(len(data[0].split(' ')))
    start=time.time()
    while(j<=len(data[0].split(' '))):
    # while(j<=500):
        pre_train=[]
        pre_sub=[]
        pre_sub_id=[]
        pre_label=torch.zeros((1,len(word2id)))
        pre_mid=math.floor((i+j)/2)
        for item in data[0].split(' ')[i:pre_mid]:
            pre_train.append(word2id[item])
            for k in range(len(item)):
                pre_sub_id.append(char2id[item[k]])
            pre_sub.append(torch.tensor(pre_sub_id))
            pre_sub_id=[]
        for item in data[0].split(' ')[pre_mid+1:j]:
            pre_train.append(word2id[item])
            for k in range(len(item)):
                pre_sub_id.append(char2id[item[k]])
            pre_sub.append(torch.tensor(pre_sub_id))
            pre_sub_id=[]
        pre_train_tensor=torch.tensor(pre_train)
        # print(pre_sub)
        pre_label[0][word2id[data[0].split(' ')[pre_mid]]]=1
        # print(pre_label)
        train.append((pre_train_tensor,pre_sub,pre_label))
        # break
        i+=1
        j+=1
    end=time.time()
    print("build trainset finish:",(end-start),"s")
    print("train length:",len(train))
    train_length=len(train)
    model=My_model()
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # for epoch in range(600):
    #     epoch_avg_loss=0
    #     start=time.time()
    #     for i in range(len(train)):
    #         # batch_start=time.time()
    #         optimizer.zero_grad()
    #         _,out=model(train[i])
    #         # print(train[i][2].shape)
    #         # print(out.shape)
    #         loss=criterion(out,train[i][2])
    #         epoch_avg_loss+=loss
    #         loss.backward()
    #         optimizer.step()
    #         # batch_end=time.time()
    #         # print("batch:",i," loss:",loss.detach().cpu().numpy()," time:",(batch_end-batch_start),"s")
    # #         # break
    #     end=time.time()
    #     epoch_avg_loss/=train_length
    #     print('epoch:',epoch,' loss:',epoch_avg_loss.detach().cpu().numpy()," time:",(end-start),"s")
    #     # break
    # torch.save(model,'mypretrain_adam_600.pth')
    pre_model = torch.load('mypretrain_adam_600.pth')
    total=0
    correct=0
    for i in range(len(train)):
        total+=1
        _,res=pre_model(train[i])
        predict=torch.argmax(res,dim=1)
        true_label=torch.argmax(train[i][2],dim=1)
        if predict==true_label:
            correct+=1
        # break
    print(total)	#4600
    print(correct)	#4457
    # epoch 600 acc:96.89%
    print("acc:",correct/total)
    
