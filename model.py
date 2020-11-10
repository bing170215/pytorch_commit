import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
import math


class code_Model(nn.Module):
    def __init__(self, hid,mark_embedding_dim,word_embedding_dim ,embedding_vocabulary_num,encode_length,n_layers):
        super(code_Model, self).__init__()
        self.hid = hid
        self.mark_embedding_dim =mark_embedding_dim
        self.word_embedding_dim=word_embedding_dim
        self.embedding_vocabulary_num=embedding_vocabulary_num
        self.encode_length = encode_length
        self.n_layers = n_layers

        '''
        开始定义网络
        '''
        #定义两个Embedding层
        self.mark_embed_layer=nn.Embedding(4,self.mark_embedding_dim)
        self.word_embedding_layer = nn.Embedding(self.embedding_vocabulary_num,self.word_embedding_dim)

        #定义代码结构经过的三层双向GRU
        self.code_structure_GRU = nn.GRU(self.word_embedding_dim+self.mark_embedding_dim,self.hid,num_layers=self.n_layers,bidirectional=False,batch_first=True,dropout=0.2)

        #定义语义序列经过的三层双向GRU
        self.Semantics_GRU = nn.GRU(self.word_embedding_dim,self.hid,num_layers=self.n_layers,bidirectional=False,batch_first=True,dropout=0.2)

        self.output = nn.Linear(2 * self.hid, self.embedding_vocabulary_num)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, mark ,word,attr):

        m_embed_en= self.mark_embed_layer(mark)
        w_embed_en = self.word_embedding_layer(word)
        # a_reshape = attr.reshape(-1,200,5)
        a_embed_en = self.word_embedding_layer(attr)

        embed_en = torch.cat((m_embed_en,w_embed_en),dim=-1)

        #经过代码结构的三层双向GRU
        code_structure_h,_ = self.code_structure_GRU(embed_en)
        #经过语义序列的三层双向GRU
        a_embed_en = a_embed_en.reshape(-1,5,self.word_embedding_dim)
        semantics_h,_ =self.Semantics_GRU(a_embed_en)
        semantics_h =semantics_h[:,-1,:].reshape(-1,self.encode_length ,self.hid)
        # Combining Code Structure and Semantics  合并前的masked_rnn_h3是 h，a_rnn_h3是z
        combine_h = torch.cat((code_structure_h,semantics_h),dim=-1)  # 合并后的masked_rnn_h3就是 h'  (？,200,64)
        code_vec = combine_h[:,-1,:]

        # code_vec = self.output(code_vec)
        #
        # code_vec = self.drop(code_vec)

        return code_vec


class commit_Model(nn.Module):
    def __init__(self, hid,word_embedding_dim ,embedding_vocabulary_num,encode_length,n_layers):
        super(commit_Model, self).__init__()
        self.hid = hid
        self.word_embedding_dim=word_embedding_dim
        self.embedding_vocabulary_num=embedding_vocabulary_num
        self.encode_length = encode_length
        self.n_layers = n_layers

        self.word_embedding_layer = nn.Embedding(self.embedding_vocabulary_num,self.word_embedding_dim)

        #定义commit经过的三层单向GRU
        self.commit_GRU = nn.GRU(self.word_embedding_dim,2*self.hid,num_layers=self.n_layers,batch_first=True,dropout=0.2)

        self.output = nn.Linear(2*self.hid,self.embedding_vocabulary_num)

        self.drop = nn.Dropout(p=0.2)

    def forward(self, msg):
        #经过commit的三层单向GRU
        embed_de = self.word_embedding_layer(msg)
        commit_h,_= self.commit_GRU(embed_de)
        commit_vec=commit_h[:,-1,:]
        # commit_vec = self.output(commit_vec)
        # commit_vec = self.drop(commit_vec)

        return commit_vec

class class_Model(nn.Module):

    def __init__(self, hid):
        super(class_Model, self).__init__()

        self.hid = hid
        self.output = nn.Linear(6*self.hid,2)
        self.drop = nn.Dropout(p=0.2)
        self.logsoftmax = nn.LogSoftmax(dim=1)



    def forward(self,code_vec,commit_vec,temperature=0.1):


        mul_vec=code_vec.mul(commit_vec)
        att_out = torch.cat((code_vec, commit_vec, mul_vec), dim=-1)
        output = self.output(att_out)
        output = self.drop(output)
        output = self.logsoftmax(output)

        #similiarity=F.cosine_similarity(code_vec,commit_vec)
        # output = similiarity/temperature
        # output=torch.sigmoid(output)
        return output




class SIMI_Model(nn.Module):


    def __init__(self, hid,mark_embedding_dim,word_embedding_dim ,embedding_vocabulary_num,encode_length,n_layers):
        super(SIMI_Model, self).__init__()
        self.hid = hid
        self.mark_embedding_dim =mark_embedding_dim
        self.word_embedding_dim=word_embedding_dim
        self.embedding_vocabulary_num=embedding_vocabulary_num
        self.encode_length = encode_length
        self.n_layers = n_layers



        '''
        开始定义网络
        '''
        #定义两个Embedding层
        self.mark_embed_layer=nn.Embedding(4,self.mark_embedding_dim)
        self.word_embedding_layer = nn.Embedding(self.embedding_vocabulary_num,self.word_embedding_dim)

        #定义代码结构经过的三层双向GRU
        self.code_structure_GRU = nn.GRU(self.word_embedding_dim+self.mark_embedding_dim,self.hid,num_layers=self.n_layers,bidirectional=False,batch_first=True,dropout=0.2)

        #定义语义序列经过的三层双向GRU
        self.Semantics_GRU = nn.GRU(self.word_embedding_dim,self.hid,num_layers=self.n_layers,bidirectional=False,batch_first=True,dropout=0.2)

        #定义commit经过的三层单向GRU
        self.commit_GRU = nn.GRU(self.word_embedding_dim,2*self.hid,num_layers=self.n_layers,batch_first=True,dropout=0.2)

        #定义一个线性层



        self.output = nn.Linear(6*self.hid,2)
        self.drop = nn.Dropout(p=0.2)

        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, mark ,word,attr,msg):

        m_embed_en= self.mark_embed_layer(mark)
        w_embed_en = self.word_embedding_layer(word)
        # a_reshape = attr.reshape(-1,200,5)
        a_embed_en = self.word_embedding_layer(attr)

        embed_en = torch.cat((m_embed_en,w_embed_en),dim=-1)





        #经过代码结构的三层双向GRU
        code_structure_h,_ = self.code_structure_GRU(embed_en)
        # print('******************************')
        # print('code_structure_h')
        # print(code_structure_h.size())
        # print('*******************************')

        #经过语义序列的三层双向GRU
        a_embed_en = a_embed_en.reshape(-1,5,self.word_embedding_dim)
        semantics_h,_ =self.Semantics_GRU(a_embed_en)
        # print('******************************')
        # print('semantics_h')
        # print(semantics_h.size())
        # print('*******************************')
        semantics_h =semantics_h[:,-1,:].reshape(-1,self.encode_length ,self.hid)

        # Combining Code Structure and Semantics  合并前的masked_rnn_h3是 h，a_rnn_h3是z
        combine_h = torch.cat((code_structure_h,semantics_h),dim=-1)  # 合并后的masked_rnn_h3就是 h'  (？,200,64)

        #经过commit的三层单向GRU
        embed_de = self.word_embedding_layer(msg)
        commit_h,_= self.commit_GRU(embed_de)
        # print('******************************')
        # print('commit_h')
        # print(commit_h.size())
        # print('*******************************')

        #经过注意力层

        code_vec = combine_h[:,-1,:]
        commit_vec=commit_h[:,-1,:]
        # print('*********************************')
        # print(code_vec)
        # print(commit_vec)
        # print(code_vec-commit_vec)
        # print('*********************************')
        mul_vec=code_vec.mul(commit_vec)
        #att_out = torch.cat((code_vec,commit_vec,code_vec-commit_vec,mul_vec),dim=-1)
        att_out = torch.cat((code_vec, commit_vec, mul_vec), dim=-1)


        output = self.output(att_out)

        output = self.drop(output)

        #print(output)
        output = self.logsoftmax(output)
        # print('*********************************')
        # print('output')
        # print(output.size())
        # print('*********************************')

        return output
