"""
glove style pretrain
"""

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from einops import rearrange

torch.manual_seed(66)


class MyDataset(Dataset):
   def __init__(self, word1,word2, labels):
       self.word1 = word1
       self.word2 = word2
       self.labels = labels
   def __len__(self):
       return len(self.word1)
   def __getitem__(self, idx):
       return self.word1[idx], self.word2[idx], self.labels[idx]
   

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        assert d_model%num_heads==0
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim = d_model//num_heads
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,1)
    def split_heads(self,x):
        batch_size,seq_len,_=x.size()
        x = x.view(batch_size,seq_len,self.num_heads,self.head_dim)
        return x.transpose(1,2)

    def Query_concat_attention(self,Q,K):
        K_tensor = rearrange(K,'b h w c -> b c w h')
        Q_chunks = torch.split(Q, 1, dim=1)
        qk_lis=[]
        for i in range(len(Q_chunks)):
            pre_qk = rearrange(Q_chunks[i], 'b h w c ->b c h w')
            qk_lis.append(torch.matmul(pre_qk,K_tensor))
        qk_total=torch.concat(qk_lis,dim=-1)
        return qk_total
    
    def forward(self,x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        pre_qk = self.Query_concat_attention(Q,K)
        pre_qk=rearrange(pre_qk,'b h w c -> b w c h')
        pre_V = rearrange(V,'b h w c-> b w c h')
        pre_qkv=torch.matmul(pre_qk,pre_V).squeeze(1)
        pre_qkv=rearrange(pre_qkv,'b w c-> b (w c)')
        res = self.W_o(pre_qkv)
        return res


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.embed = nn.Embedding(1552,512)
        self.mhqc1 = MultiHeadAttention(d_model=512,num_heads=8)
        self.mhqc2 = MultiHeadAttention(d_model=512,num_heads=8)
        self.tanh = nn.Tanh()

    def forward(self,word1,word2,labels):
        input1=self.embed(word1)
        input2=self.embed(word2)
        mhqc_1=self.mhqc1(input1)
        mhqc_2=self.mhqc2(input2)
        res = torch.log(torch.abs(mhqc_1*mhqc_2))**2
        pre_labels=torch.log(labels.unsqueeze(1))**2
        final_res=self.tanh(pre_labels-res)
        final_res = torch.mean(final_res)
        return final_res

if __name__=='__main__':
    with open("train.txt",'r') as f:
        train_data = f.readlines()
    with open("train.txt",'r') as f:
        test_data = f.readlines()
    with open("corpus",'r') as f:
        data = f.readlines()
    
    word2id=np.load('word2id.npy', allow_pickle=True).item()
    id2word=np.load('id2word.npy', allow_pickle=True).item()
    print(len(word2id))
    train_word1_numpy=np.load('train_word1.npy')
    train_word2_numpy=np.load('train_word2.npy')
    train_labels=np.load('train_labels.npy')
    train_word1_tensor=torch.from_numpy(train_word1_numpy).long().unsqueeze(1)
    train_word2_tensor=torch.from_numpy(train_word2_numpy).long().unsqueeze(1)
    train_labels_tensor=torch.from_numpy(train_labels)

    dataset = MyDataset(train_word1_tensor, train_word2_tensor,train_labels_tensor)
    print(len(dataset))
    model=My_model()
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    for epoch in range(20):
        start=time.time()
        epoch_avg_loss=0
        count=0
        for batch_word1, batch_word2, batch_labels in dataloader:
            optimizer.zero_grad()
            res=model(batch_word1,batch_word2,batch_labels).float()
            res.backward()
            optimizer.step()
            epoch_avg_loss+=res
            count+=1
        epoch_avg_loss/=count
        end=time.time()
        print("epoch:",epoch," loss:",epoch_avg_loss.detach().numpy(),"time:",end-start,"s")
        if epoch_avg_loss<0:
            break
        # break

    

    