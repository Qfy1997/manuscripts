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

    def scaled_dot_product_attention(self,Q,K,V):
        scores = torch.matmul(Q,K.transpose(-2,-1))
        scores = scores/torch.sqrt(torch.tensor(self.head_dim,dtype=torch.float32))
        attention_weights = F.softmax(scores,dim=-1)
        output = torch.matmul(attention_weights,V)
        return output,attention_weights
    
    def forward(self,x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        attention_output,attention_weights = self.scaled_dot_product_attention(Q,K,V)
        attention_output = attention_output.transpose(1,2)
        batch_size,seq_len,_,_ = attention_output.size()
        concat_output = attention_output.contiguous().view(batch_size,seq_len,self.d_model)
        output = self.W_o(concat_output)
        return output,attention_weights


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.embed = nn.Embedding(1552,512)
        self.mha1 = MultiHeadAttention(d_model=512,num_heads=8)
        self.mha2 = MultiHeadAttention(d_model=512,num_heads=8)
        self.tanh = nn.Tanh()

    def forward(self,word1,word2,labels):
        input1=self.embed(word1)
        input2=self.embed(word2)
        mha_1,_=self.mha1(input1)
        mha_2,_=self.mha2(input2)
        mha1=mha_1.squeeze(2)
        mha2=mha_2.squeeze(2)
        res = torch.log(torch.abs(mha1+mha2))**2
        pre_labels=torch.log(labels.unsqueeze(1))**2
        final_res=self.tanh((pre_labels-res))
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
    # print(len(word2id))
    train_word1_numpy=np.load('train_word1.npy')
    train_word2_numpy=np.load('train_word2.npy')
    # First, segment each sentence in the corpus.
    # For each sentence, compute the edit distance between the first word of the sentence and all other words within the current sentence.
    # The final two words correspond to a label equal to the multiplication of their edit distance and inter-sentence distance.
    train_labels=np.load('train_labels.npy')
    train_word1_tensor=torch.from_numpy(train_word1_numpy).long().unsqueeze(1)
    train_word2_tensor=torch.from_numpy(train_word2_numpy).long().unsqueeze(1)
    train_labels_tensor=torch.from_numpy(train_labels).float()
    dataset = MyDataset(train_word1_tensor, train_word2_tensor,train_labels_tensor)
    # print(len(dataset))
    model=My_model()
    optimizer = optim.SGD(model.parameters(),lr=0.001)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # print(len(dataloader))
    # for batch_word1, batch_word2, batch_labels in dataloader:
    #         res=model(batch_word1,batch_word2,batch_labels)
    #         print(res)
    #         break
    for epoch in range(600):
        start=time.time()
        epoch_avg_loss=0
        count=0
        for batch_word1, batch_word2, batch_labels in dataloader:
            optimizer.zero_grad()
            # print(batch_word1.shape, batch_word2.shape,batch_labels.shape)
            res=model(batch_word1,batch_word2,batch_labels).float()
            # print("res:",res)
            count+=1
            # print(batch_labels)
            # loss = criterion(res,batch_labels)
            res.backward()
            optimizer.step()
            epoch_avg_loss+=res
            # break
        epoch_avg_loss/=count
        end=time.time()
        print("epoch:",epoch," loss:",epoch_avg_loss,"time:",end-start,"s")
        if epoch_avg_loss<0:
            break
        # break
        

    

    