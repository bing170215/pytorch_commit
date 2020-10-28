from data4CopynetV3 import Data4CopynetV3
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

def trans_data(t1,t2,t3,t4,end):


    y_array = np.zeros([len(t1),])
    y_array[:end]=1       #[1,1,1;0,0,0;0,0,0;0,0,0]
    #yy = np.array([one_hot(int(x), 2) for x in y_array])
    yy =y_array

    sf = np.arange(len(t1))
    np.random.shuffle(sf)

    tt1, tt2, tt3, tt4 ,yy= t1[sf], t2[sf], t3[sf], t4[sf],yy[sf]
    #tt1, tt2, tt3, tt4, y =tt1[:end], tt2[:end], tt3[:end], tt4[:end] ,yy[:end]
    return tt1,tt2,tt3,tt4,yy


def one_hot(value,classes):
    one_hots = np.zeros([classes,])
    one_hots[value] = 1
    return one_hots

def load_data(d_mark, d_word, d_attr, mg,TR_E,VA_E, batch_size=100):

    #t1, t2, t3, t4, y = trans_data( t1, t2, t3, t4,VA_E)
    x1 = torch.from_numpy(d_mark)
    x2 = torch.from_numpy(d_word)
    x3 = torch.from_numpy(d_attr)
    x4 = torch.from_numpy(mg)

    # put into tensor dataset
    train_data = TensorDataset(x1[:TR_E], x2[:TR_E],x3[:TR_E],x4[:TR_E])
    val_data = TensorDataset(x1[TR_E:VA_E],x2[TR_E:VA_E],x3[TR_E:VA_E],x4[TR_E:VA_E])


    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size, drop_last=False)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size, drop_last=False)


    return train_data_loader, valid_data_loader


def cal_accu(output,target):
    output = output.cpu()
    target = target.cpu()
    right =0
    for i in range(output.size(0)):
        if output[i][0]>=output[i][1]:
            if target[i]==0:
                right +=1
            else:
                continue
        else:
            if output[i][0]<output[i][1]:
                if target[i] ==1:
                    right +=1
                else:
                    continue

    return right/output.size(0)


def cal_accu2(output,target):
    output = output.cpu()
    target = target.cpu()
    right =0
    for i in range(output.size(0)):
        if output[i]<=0.5:
            if target[i]==0:
                right +=1
            else:
                continue
        else:
            if output[i]>0.5:
                if target[i] ==1:
                    right +=1
                else:
                    continue

    return right/output.size(0)



def evaluator(output,target):

    out = torch.zeros(output.size(0))
    output = output.cpu()
    target = target.cpu()
    for i in range(output.size(0)):
        if output[i,0]>=output[i,1]:
            out[i]=0
        else:
            out[i] = 1
    # for i in range(output.size(0)):
    #     if output[i]<0.5:
    #         out[i]=0
    #     else:
    #         out[i] = 1
    # err = torch.sum(torch.abs(output[:,:,1]-target))
    # err = err.item()/output.size(0)

    err = torch.sum(torch.abs(out-target))

    err = err.item()/out.size(0)

    fpr, tpr, threshold = roc_curve(target.numpy(), out.numpy())
    roc_auc = auc(fpr, tpr)
    tp, tn, fp, fn = calc_tptnfpfn_dyn(out,target)
    tp = tp/output.size(0)
    tn = tn/output.size(0)
    fp = fp/output.size(0)
    fn = fn/output.size(0)
    tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp, tn, fp, fn)
    return err, tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc

def evaluator2(output,target):

    out = torch.zeros(output.size(0))
    output = output.cpu()
    target = target.cpu()
    for i in range(output.size(0)):
        if output[i]<=0.5:
            out[i]=0
        else:
            out[i] = 1
    # err = torch.sum(torch.abs(output[:,:,1]-target))
    # err = err.item()/output.size(0)

    err = torch.sum(torch.abs(out-target))

    err = err.item()/out.size(0)

    fpr, tpr, threshold = roc_curve(target.numpy(), out.numpy())
    roc_auc = auc(fpr, tpr)
    tp, tn, fp, fn = calc_tptnfpfn_dyn(out,target)
    tp = tp/output.size(0)
    tn = tn/output.size(0)
    fp = fp/output.size(0)
    fn = fn/output.size(0)
    tpr, fpr, tnr, fnr, f1_score = evaluation_indicator(tp, tn, fp, fn)
    return err, tp, tn, fp, fn, tpr, fpr, tnr, fnr, f1_score, roc_auc





def evaluation_indicator(tp,tn,fp,fn):
    try:
        tpr = float(tp) / (tp + fn)
    except ZeroDivisionError:
        tpr=0
    try:
        fpr = float(fp) / (fp + tn)
    except ZeroDivisionError:
        fpr = 0
    try:
        tnr = float(tn) / (tn + fp)
    except ZeroDivisionError:
        tnr = 0
    try:
        fnr = float(fn) / (tp + fn)
    except ZeroDivisionError:
        fnr = 0

    try:
        p = tp / (tp + fp)
        r = tp / (tp + fn)

        f1_score = 2 * p * r / (p + r)
    except ZeroDivisionError:
        f1_score = 0
    return tpr, fpr, tnr, fnr, f1_score

def calc_tptnfpfn_dyn(out,target):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(out.shape[0]):
        if target[i] == 1:
            # positive
            if out[i] == 1:
                # true positive
                tp += 1
            else:
                # false positive
                fn += 1
        else:
            # negative
            if out[i] == 1:
                # true negative
                fp += 1
            else:
                # false neg
                tn += 1
    return tp,tn,fp,fn

