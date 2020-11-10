import torch
import time
import random
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, auc
import sys
from model import *
from tools import *
from vec import *

writer = SummaryWriter()

#定义一些必要的参数
TR_S = 0                        # train_start_index #训练集开始时的索引，
TR_E = 75000                   # train_end_index #训练集结束时的索引 75000
VA_S = 75000                  # valid_start_index #验证集开始时的索引75000
VA_E = 83000                 # valid_end_index #验证集结束时的索引83000
# TE_S = 83000                   # test_start_index #测试集开始时的索引83000
# TE_E = 90661                    # test_end_index #测试集结束时的索引
TR_BS = 64                  # train batch size #训练时的batch尺寸
EP = 500                        # trian epoch  #训练的epoch
TE_BS = 1                       # test batch size #测试的epoch
E_L = 200                       # encoder len #编码部分结构序列的最大长度
A_N = 5                         # attribute number #语义序列的最大长度
D_L = 20                        # decoder len #解码部分commit信息的长度被设置为20
EM_V = 24634                    # embedding vocabulary num #embedding词汇的数量
DE_V = 10130                    # decoder vocabulary num #解码部分词汇的数量
SEED = 1                        # random seed #随机种子
MED = 50                        # mark embedding dim #符号的嵌入维度
WED = 150   #150                # TODO: word embedding dim#单词的嵌入维度被设置为150
HS = 256                     # TODO: hidden size 隐藏层的尺寸
ATN = 64                        # attention num
TR_DR = 0.1                     # drop rate for train
TE_DR = 0.                      # drop rate for test
PC = 2                          # patience
BM_S = 2                        # beam size
VER = 12                        # data version
NEG =5                        #负样本比正样本的比例       调整正负样本比例
LR = 0.001
LR_code = 0.001
LR_commit=0.001
LR_class = 0.001
GRU_layer=2



torch.cuda.set_device(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
code_learner = code_Model(hid=HS,mark_embedding_dim= MED,word_embedding_dim =WED,embedding_vocabulary_num = EM_V ,encode_length=E_L ,n_layers=GRU_layer).to(device)
commit_learner = commit_Model(hid=HS,word_embedding_dim =WED,embedding_vocabulary_num = EM_V ,encode_length=E_L ,n_layers=GRU_layer).to(device)
class_learner = class_Model(hid=HS,).to(device)

op_code=optim.Adam(code_learner.parameters(), lr=LR_code)
op_commit=optim.Adam(commit_learner.parameters(), lr=LR_commit)
op_class=optim.Adam(class_learner.parameters(), lr=LR_class)





#simi_learner= SIMI_Model(hid=HS,mark_embedding_dim= MED,word_embedding_dim =WED,embedding_vocabulary_num = EM_V ,encode_length=E_L ,n_layers=GRU_layer).to(device)
# optimizer
#op_simi = optim.Adam(simi_learner.parameters(), lr=LR)

print('正在从磁盘中读取数据')
dataset = Data4CopynetV3()
dataset.load_data(VER)  # 将data version传进去，从磁盘中加载数据
# 传进去三个参数，分别是 训练集索引的开始，验证集索引的结束，解码部分的词汇数量
# d_mark[83000,200], d_word[83000,200], d_attr[83000,200,5], mg[83000,21], genmask[10130     ], copymask[10130    ]
d_mark, d_word, d_attr, mg, genmask, copymask = dataset.gen_tensor_negative2(TR_S, VA_E, DE_V, diff_len=E_L)

train_loader, val_loader, = load_data(d_mark, d_word, d_attr, mg,TR_E,VA_E,batch_size=TR_BS )

print('数据读取完成')

loss_fn = torch.nn.NLLLoss()


def sample_negtivate(pos_target,flag=False):
    batch_size = pos_target.shape[0]

    #验证集
    if flag==True:
        #按照相同的概率选取每个负样例的代码段
        cur_p=np.ones(VA_E-TR_E)*(1/(VA_E-TR_E))
        choosen_list = np.arange(TR_E,VA_E,1)
    else:
        cur_p = np.ones(TR_E)*(1/(TR_E))
        choosen_list=np.arange(0,TR_E,1)



    # #存储负样例
    # negative_marks=np.zeros((batch_size,NEG,E_L))
    # negative_words = np.zeros((batch_size, NEG,E_L))
    # negative_attrs = np.zeros((batch_size, NEG,E_L,A_N))


    #cur_p = p.copy()
    # target_idx = pos_target[i]
    # cur_p[target_idx]=0
    negative_sample_idx=np.random.choice(choosen_list,size=(batch_size,NEG),replace=False,p=cur_p)
    #print(negative_sample_idx.shape)
    # negative_marks[i,:,:]=np.array(d_mark)[negative_sample_idx]
    # negative_words[i, :,:] = np.array(d_word)[negative_sample_idx]
    # negative_attrs[i, :,:,:] = np.array(d_attr)[negative_sample_idx]
    negative_marks=np.array(d_mark)[negative_sample_idx]
    negative_words = np.array(d_word)[negative_sample_idx]
    negative_attrs = np.array(d_attr)[negative_sample_idx]

    negative_marks=torch.from_numpy(negative_marks)
    negative_words=torch.from_numpy(negative_words)
    negative_attrs=torch.from_numpy(negative_attrs)

    commit_msgs=np.array(dataset.msg)[pos_target]
    negative_commit = dataset.get_native_mg(commit_msgs,negative_sample_idx, )
    negative_commit = torch.from_numpy(negative_commit)

    return negative_marks,negative_words,negative_attrs,negative_commit









def train_Similarity(epoch):
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
    code_learner.train()
    commit_learner.train()
    class_learner.train()
    code_vecs=None
    for batch_idx ,data in enumerate(train_loader):
        if batch_idx % 100 == 0:
            print('batch idx:', batch_idx)
        mark = data[0].long().to(device)
        word = data[1].long().to(device)
        attr =data[2].long().to(device)
        msg = data[3].long().to(device)

        pos_idx = np.arange(batch_idx * TR_BS, batch_idx * TR_BS + mark.size(0), 1)
        negative_marks, negative_words, negative_attrs, negative_commits = sample_negtivate(pos_idx)

        # 正样本标签
        pos_label = torch.ones((mark.size(0))).long().to(device)
        # 负样本的标签
        neg_label = torch.zeros((mark.size(0))).long().to(device)
        #全1矩阵
        ones = torch.ones((mark.size(0))).to(device)

        #op_simi.zero_grad()
        op_code.zero_grad()
        op_commit.zero_grad()
        op_class.zero_grad()

        #正例的正向传播
        #y_hat = simi_learner(mark,word,attr,msg)
        pos_code_vec = code_learner(mark,word,attr)
        code_vecs = np.concatenate((code_vecs, pos_code_vec.detach().cpu().numpy()),
                                   axis=0) if code_vecs is not None else pos_code_vec.detach().cpu().numpy()

        commit_vec = commit_learner(msg)
        pos_score = class_learner(pos_code_vec,commit_vec)
        loss = loss_fn(pos_score, pos_label.long())
        # print(pos_score.size())
        #loss = pos_score.log()

        #负例的正向传播
        for i in range(NEG):

            negative_mark = negative_marks[:, i,:].long().to(device)
            negative_word = negative_words[:, i,:].long().to(device)
            negative_attr = negative_attrs[:,i,:,:].long().to(device)
            negative_commit = negative_commits[:,i,:].long().to(device)
            neg_code_vec = code_learner(negative_mark,negative_word,negative_attr)
            neg_commit_vec = commit_learner(negative_commit)

            neg_score = class_learner(neg_code_vec,neg_commit_vec)
            loss +=loss_fn(neg_score,neg_label.long())/NEG

        #loss = -loss.mean()

        # backward and optimize
        loss.backward()
        op_code.step()
        op_commit.step()
        op_class.step()

        #op_simi.step()
        accu = cal_accu(pos_score, pos_label.long())
        err,tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(pos_score, pos_label.long())
        if batch_idx %100==0:

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
    print('当前epoch为'+str(epoch))
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
    print('计算top-k准确度！！！')
    with torch.no_grad():
        code_learner.eval()
        commit_learner.eval()
        class_learner.eval()

        num_of_choosen = 100
        choosen_idx = np.random.choice(TR_E, size=num_of_choosen, replace=False)

        correct1 = 0
        correct2 = 0
        correct3 = 0


        for idx in choosen_idx:
            commit = split_msg(dataset.msgtext[idx])
            # top5_ids = get_top_k(dataset, commit[0], 5, code_vecs, commit_learner, class_learner,device,DATA_SIZE=TR_E)
            # top10_ids = get_top_k(dataset, commit[0], 10, code_vecs, commit_learner, class_learner,device,DATA_SIZE=TR_E)
            top20_ids = get_top_k(dataset, commit[0], 20, code_vecs, commit_learner, class_learner,device,DATA_SIZE=TR_E)
            top5_ids=top20_ids[:5]
            top10_ids = top20_ids[:10]
            if idx in top5_ids:
                correct1 = correct1 +1

            if idx in top10_ids:
                correct2 = correct2 +1

            if idx in top20_ids:
                correct3 = correct3 +1

        accu5 = float(correct1 / num_of_choosen)
        accu10 = float(correct2 / num_of_choosen)
        accu20 = float(correct3 / num_of_choosen)
        print('******************************')
        print('topK=' + str(5))
        print('命中数=' + str(correct1))
        print('准确率=' + str(accu5))
        print('******************************')
        print('topK=' + str(10))
        print('命中数=' + str(correct2))
        print('准确率=' + str(accu10))
        print('******************************')
        print('topK=' + str(20))
        print('命中数=' + str(correct3))
        print('准确率=' + str(accu20))

        writer.add_scalar('train_loss', np.mean(loss_batch), global_step=epoch)
        writer.add_scalar('train_top5', accu5, global_step=epoch)
        writer.add_scalar('train_top10', accu10, global_step=epoch)
        writer.add_scalar('train_top20', accu20, global_step=epoch)








def val_Similarity(epoch):
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
    code_vecs=None
    for batch_idx ,data in enumerate(val_loader):
        if batch_idx % 100 == 0:
            print('batch idx:', batch_idx)
        data = data
        mark = data[0].long().to(device)
        word = data[1].long().to(device)
        attr =data[2].long().to(device)
        msg = data[3].long().to(device)

        pos_idx = np.arange(VA_S+batch_idx * TR_BS, VA_S+batch_idx * TR_BS + mark.size(0), 1)
        negative_marks, negative_words, negative_attrs,negative_commits = sample_negtivate(pos_idx,flag=True )

        # 正样本标签
        pos_label = torch.ones((mark.size(0))).long().to(device)
        # 负样本的标签
        neg_label = torch.zeros((mark.size(0))).long().to(device)
        #全1矩阵
        ones = torch.ones((mark.size(0))).to(device)
        code_learner.eval()
        commit_learner.eval()
        class_learner.eval()
        # 正例的正向传播
        # y_hat = simi_learner(mark,word,attr,msg)
        pos_code_vec = code_learner(mark, word, attr)
        code_vecs = np.concatenate((code_vecs, pos_code_vec.detach().cpu().numpy()),
                                   axis=0) if code_vecs is not None else pos_code_vec.detach().cpu().numpy()
        commit_vec = commit_learner(msg)
        pos_score = class_learner(pos_code_vec, commit_vec)
        loss = loss_fn(pos_score, pos_label.long())
        #loss = pos_score.log()

        # 负例的正向传播
        for i in range(NEG):
            negative_mark = negative_marks[:, i,:].long().to(device)
            negative_word = negative_words[:, i,:].long().to(device)
            negative_attr = negative_attrs[:,i,:,:].long().to(device)
            negative_commit = negative_commits[:,i,:].long().to(device)
            neg_code_vec = code_learner(negative_mark,negative_word,negative_attr)
            neg_commit_vec = commit_learner(negative_commit)

            neg_score = class_learner(neg_code_vec,neg_commit_vec)
            loss +=loss_fn(neg_score,neg_label.long())/NEG
            #loss +=(ones-neg_score).log()/NEG

        #loss=-loss.mean()

        # backward and optimize
        accu = cal_accu(pos_score, pos_label.long())
        err,tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc = evaluator(pos_score, pos_label.long())
        if batch_idx%100==0:

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


    print('计算top-k准确度！！！')
    with torch.no_grad():
        num_of_choosen = 100
        choosen_idx = np.random.choice(np.arange(TR_E,VA_E), size=num_of_choosen, replace=False)

        correct1 = 0
        correct2 = 0
        correct3 = 0

        for idx in choosen_idx:
            commit = split_msg(dataset.msgtext[idx])
            # top5_ids = get_top_k(dataset, commit[0], 5, code_vecs, commit_learner, class_learner,device, cur=TR_E,DATA_SIZE=VA_E)
            # top10_ids = get_top_k(dataset, commit[0], 10, code_vecs, commit_learner, class_learner,device, cur=TR_E, DATA_SIZE=VA_E)
            top20 = get_top_k(dataset, commit[0], 20, code_vecs, commit_learner, class_learner, device,cur=TR_E, DATA_SIZE=VA_E)
            top20_ids=[i+VA_S for i in top20]
            top5_ids=top20_ids[:5]
            top10_ids = top20_ids[:10]
            if idx in top5_ids:
                correct1 = correct1 + 1

            if idx in top10_ids:
                correct2 = correct2 + 1

            if idx in top20_ids:
                correct3 = correct3 + 1

        accu5 = float(correct1 / num_of_choosen)
        accu10 = float(correct2 / num_of_choosen)
        accu20 = float(correct3 / num_of_choosen)
        print('******************************')
        print('topK=' + str(5))
        print('命中数=' + str(correct1))
        print('准确率=' + str(accu5))
        print('******************************')
        print('topK=' + str(10))
        print('命中数=' + str(correct2))
        print('准确率=' + str(accu10))
        print('******************************')
        print('topK=' + str(20))
        print('命中数=' + str(correct3))
        print('准确率=' + str(accu20))

        writer.add_scalar('val_loss', np.mean(loss_batch), global_step=epoch)
        writer.add_scalar('val_top5', accu5, global_step=epoch)
        writer.add_scalar('val_top10', accu10, global_step=epoch)
        writer.add_scalar('val_top20', accu20, global_step=epoch)
    return np.mean(loss_batch)

loss=1000
main_path = './models/'
code_path = main_path + 'code_Model_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl'
commit_path = main_path + 'commit_Model_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl'
class_path = main_path + 'class_Model_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl'
for epoch in range(EP):
    print('current_epoch:'+str(epoch))

    train_Similarity(epoch)
    print('*****************进行验证********************')
    val_loss=val_Similarity(epoch)
    if val_loss!=np.nan and val_loss<loss:
        loss = val_loss
        print('best epoch:', epoch)
        print('正在保存模型')
        torch.save(code_learner, code_path)
        torch.save(commit_learner, commit_path)
        torch.save(class_learner, class_path)


    torch.save(code_learner, main_path + 'code_Model_end_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl')
    torch.save(commit_learner, main_path + 'commit_Model_end_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl')
    torch.save(class_learner, main_path + 'class_Model_end_NEG'+str(NEG)+'_nlloss_neg_hs512.pkl')






print('*******************************************')
print('*******************************************')
print('训练结束！！！')
print('*******************************************')
print('*******************************************')