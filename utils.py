import os.path as osp
import numpy as np
import math
import csv, json
import scipy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
from scipy.spatial.distance import cdist
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.io import read_txt_array
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


def cal_acc(data, netF, netC, conv_time=30):
    with torch.no_grad():
        data_f = netF(data.x, data.edge_index, conv_time)
        output = netC(data_f, data.edge_index)
        
    output = F.log_softmax(output, dim=1)
    loss = F.nll_loss(output, data.y)
    pred = output.max(dim=1)[1]
    
    correct = pred.eq(data.y).sum().item()
    acc = correct * 1.0 / len(data.y)

    pred = pred.cpu().numpy()
    gt = data.y.cpu().numpy()
    macro_f1 = f1_score(gt, pred, average='macro')
    micro_f1 = f1_score(gt, pred, average='micro')

    return acc, macro_f1, micro_f1, loss

def guassian_kernel(source, target, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):

    n_samples = int(source.size()[0]) + int(target.size()[0])  
    total = torch.cat([source, target], dim=0) 
    total0 = total.unsqueeze(0).expand(int(total.size(0)), 
                                       int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), 
                                       int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source_feat, target_feat, sampling_num = 1000, times = 5):
    source_num = source_feat.size(0)
    target_num = target_feat.size(0)

    source_sample = torch.randint(source_num, (times, sampling_num))
    target_sample = torch.randint(target_num, (times, sampling_num))

    mmd = 0
    for i in range(times):
        source_sample_feat = source_feat[source_sample[i]]
        target_sample_feat = target_feat[target_sample[i]]

        mmd = mmd + get_MMD(source_sample_feat, target_sample_feat)

    mmd = mmd / times
    return mmd


def get_MMD(source_feat, target_feat, kernel_mul=2.0, kernel_num=5
            , fix_sigma=None):
    kernels = guassian_kernel(source_feat, 
                              target_feat,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num,
                              fix_sigma=fix_sigma)
    
    batch_size = min(int(source_feat.size()[0]), int(target_feat.size()[0]))  
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss
