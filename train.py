import torch
import time
import random
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim

from sklearn.metrics import roc_curve, auc
import sys
from model import *
from tools import *


#定义一些必要的参数
TR_S = 0                        # train_start_index #训练集开始时的索引，
TR_E = 75000                    # train_end_index #训练集结束时的索引
VA_S = 75000                    # valid_start_index #验证集开始时的索引
VA_E = 83000                    # valid_end_index #验证集结束时的索引
TE_S = 83000                    # test_start_index #测试集开始时的索引
TE_E = 90661                    # test_end_index #测试集结束时的索引
TR_BS = 100                    # train batch size #训练时的batch尺寸
EP = 50                         # trian epoch  #训练的epoch
TE_BS = 1                       # test batch size #测试的epoch
E_L = 200                       # encoder len #编码部分结构序列的最大长度
A_N = 5                         # attribute number #语义序列的最大长度
D_L = 20                        # decoder len #解码部分commit信息的长度被设置为20
EM_V = 24634                    # embedding vocabulary num #embedding词汇的数量
DE_V = 10130                    # decoder vocabulary num #解码部分词汇的数量
SEED = 1                        # random seed #随机种子
MED = 50                        # mark embedding dim #符号的嵌入维度
WED = 150   #150                       # TODO: word embedding dim#单词的嵌入维度被设置为150
HS = 128                       # TODO: hidden size 隐藏层的尺寸
ATN = 64                        # attention num
TR_DR = 0.1                     # drop rate for train
TE_DR = 0.                      # drop rate for test
PC = 2                          # patience
BM_S = 2                        # beam size
VER = 12                        # data version
NEG =1                          #负样本比正样本的比例       调整正负样本比例
LR = 0.001
GRU_layer=2


print('LR:',LR)
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simi_learner= SIMI_Model(hid=HS,mark_embedding_dim= MED,word_embedding_dim =WED,embedding_vocabulary_num = EM_V ,encode_length=E_L ,n_layers=GRU_layer).to(device)
# optimizer
op_simi = optim.Adam(simi_learner.parameters(), lr=LR)

train_loader, val_loader, = load_data(TR_S,TR_E,VA_E,DE_V,NEG,VER,E_L,batch_size=TR_BS )
loss_fn = torch.nn.NLLLoss()

def train_Similarity(e):
    loss_batch = []
    accus_batch = []
    err_batch = []
    tp_batch =[]
    tn_batch=[]
    fp_batch = []
    fn_batch =[]
    tpr_batch=[]
    fpr_batch =[]
    tnr_batch=[]
    fnr_batch=[]
    f1_score_batch=[]
    roc_auc_batch=[]

    for batch_idx ,data in enumerate(train_loader):
        print('batch idx:', batch_idx)
        data = data
        mark = data[0].long().to(device)
        word = data[1].long().to(device)
        attr =data[2].long().to(device)
        msg = data[3].long().to(device)
        y= data[4].long().to(device)


        op_simi.zero_grad()
        y_hat = simi_learner(mark,word,attr,msg)

        loss = loss_fn(y_hat, y.long())
        #loss = torch.mean(torch.abs(y_hat - y))
        #loss = torch.mean(torch.abs(y_hat - y.long()))

        # backward and optimize
        loss.backward()
        op_simi.step()
        accu = cal_accu(y_hat, y.long())
        err,tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(y_hat, y.long())

        print('loss:'+str(loss.item())+'acc:'+str(accu)+'auc:'+str(roc_auc))

        loss_batch.append(loss.item())
        accus_batch.append(accu)
        err_batch.append(err)
        tp_batch.append(tp)
        tn_batch.append(tn)
        fp_batch.append(fp)
        fn_batch.append(fn)
        tpr_batch.append(tpr)
        fpr_batch.append(fpr)
        tnr_batch.append(tnr)
        fnr_batch.append(fnr)
        f1_score_batch.append(f1_score)
        roc_auc_batch.append(roc_auc)

    print('********************************')
    print('当前epoch执行完毕')
    print('当前epoch为'+str(e))
    print('********************************')
    print('********************************')
    print('loss:',np.mean(loss_batch))
    print('accu:',np.mean(accus_batch))
    print('err:', np.mean(err_batch))
    print('tp:', np.mean(tp_batch))
    print('tn:', np.mean(tn_batch))
    print('fp:', np.mean(fp_batch))
    print('fn:', np.mean(fn_batch))
    print('tpr:', np.mean(tpr_batch))
    print('fpr:', np.mean(fpr_batch))
    print('tnr:', np.mean(tnr_batch))
    print('fnr:', np.mean(fnr_batch))
    print('f1:', np.mean(f1_score_batch))
    print('auc:', np.mean(roc_auc_batch))
    print('********************************')


def val_Similarity():
    loss_batch = []
    accus_batch = []
    err_batch = []
    tp_batch =[]
    tn_batch=[]
    fp_batch = []
    fn_batch =[]
    tpr_batch=[]
    fpr_batch =[]
    tnr_batch=[]
    fnr_batch=[]
    f1_score_batch=[]
    roc_auc_batch=[]

    for batch_idx ,data in enumerate(val_loader):
        print('batch idx:', batch_idx)
        data = data
        mark = data[0].long().to(device)
        word = data[1].long().to(device)
        attr =data[2].long().to(device)
        msg = data[3].long().to(device)
        y= data[4].long().to(device)


        y_hat = simi_learner(mark,word,attr,msg)

        loss = loss_fn(y_hat, y.long())
        #loss = torch.mean(torch.abs(y_hat - y.long()))

        # backward and optimize

        accu = cal_accu(y_hat, y.long())
        err,tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(y_hat, y.long())

        print('val_loss:'+str(loss.item())+'val_acc:'+str(accu)+'val_auc:'+str(roc_auc))

        loss_batch.append(loss.item())
        accus_batch.append(accu)
        err_batch.append(err)
        tp_batch.append(tp)
        tn_batch.append(tn)
        fp_batch.append(fp)
        fn_batch.append(fn)
        tpr_batch.append(tpr)
        fpr_batch.append(fpr)
        tnr_batch.append(tnr)
        fnr_batch.append(fnr)
        f1_score_batch.append(f1_score)
        roc_auc_batch.append(roc_auc)

    print('********************************')
    print('验证集指标')
    print('********************************')
    print('********************************')
    print('val_loss:',np.mean(loss_batch))
    print('val_accu:',np.mean(accus_batch))
    print('val_err:', np.mean(err_batch))
    print('val_tp:', np.mean(tp_batch))
    print('val_tn:', np.mean(tn_batch))
    print('val_fp:', np.mean(fp_batch))
    print('val_fn:', np.mean(fn_batch))
    print('val_tpr:', np.mean(tpr_batch))
    print('val_fpr:', np.mean(fpr_batch))
    print('val_tnr:', np.mean(tnr_batch))
    print('val_fnr:', np.mean(fnr_batch))
    print('val_f1:', np.mean(f1_score_batch))
    print('val_auc:', np.mean(roc_auc_batch))
    print('********************************')

for epoch in range(EP):
    print('current_epoch:'+str(epoch))

    train_Similarity(epoch)
    print('*****************进行验证********************')
    val_Similarity()

print('*******************************************')
print('*******************************************')
print('训练结束！！！')
print('*******************************************')
print('*******************************************')