import time
import argparse
import os, sys
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.transforms import Constant
from network import *
from utils import MMD, cal_acc
from pygda.datasets import CitationDataset, TwitchDataset
from pygda.datasets import AirportDataset
import copy
from torch.nn.functional import normalize
import random
from scipy.spatial.distance import cdist
from torch_geometric.utils import degree, is_undirected, to_undirected
from torch_geometric.transforms import OneHotDegree

def print_args(args):
    s = '=======================================================================================\n'
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def load_data(args, path_src, path_tgt):
    if args.domain == 'Citation': # DBLPv7   Citationv1   ACMv9
        source_dataset = CitationDataset(path_src, args.source)
        target_dataset = CitationDataset(path_tgt, args.target)
        na_source = args.source[0]
        na_target = args.target[0]
        if args.source == 'ACMv9' and args.target == 'Citationv1':
            args.lr = 0.005
            args.tgt_prop = 5
            args.weight = 5

    elif args.domain == 'Twitch':
        args.bottleneck = 64
        args.tgt_prop = 1
        source_dataset = TwitchDataset(path_src, args.source)
        target_dataset = TwitchDataset(path_tgt, args.target)
        na_source = args.source
        na_target = args.target

    elif args.domain == 'Airport':
        source_dataset = AirportDataset(path_src, args.source)
        target_dataset = AirportDataset(path_tgt, args.target)
        na_source = args.source[0]
        na_target = args.target[0]
        max_degree = 0
        for name in {'BRAZIL', 'USA', 'EUROPE'}:
            path = osp.join(osp.dirname(osp.realpath(__file__)), 'data/', args.domain, name)
            dataset = AirportDataset(path, name)
            data_degree = max(degree(dataset[0].edge_index[0,:]))
            if data_degree > max_degree:
                max_degree = data_degree

        target_dataset.transform = OneHotDegree(int(max_degree))
        source_dataset.transform = OneHotDegree(int(max_degree))

    return source_dataset, target_dataset, na_source, na_target

if __name__ == '__main__':
    # model agnostic params
    parser = argparse.ArgumentParser(description='TASSC')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--bottleneck', type=int, default=128, help='hidden size')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--times', type=int, default=5, help='run times')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')

    parser.add_argument('--net', type=str, default='base', help="basemodel")
    parser.add_argument('--domain', type=str, default='Citation', help='Citation, Twitch, Airport')
    parser.add_argument('--source', type=str, default='ACMv9', help='source domain data: DBLPv7  Citationv1   ACMv9 EUROPE USA')
    parser.add_argument('--target', type=str, default='Citationv1', help='target domain data: DBLPv7   Citationv1   ACMv9')
    parser.add_argument('--num_classes', type=int, default=None, help='class number')
    parser.add_argument('--num_features', type=int, default=None, help='original feature dimension')

    parser.add_argument('--domain_enhance', type=int, default=1, help='1: t_enhance; 2: s_enhance')
    parser.add_argument('--cons_par', type=float, default=1.0, help='constrative loss parameter')
    parser.add_argument('--tau', type=float, default=0.25, help='temp parameter')
    parser.add_argument('--weight', type=float, default=10, help='trade-off parameter')
    parser.add_argument('--src_prop', type=int, default=0, help='the number of propagation layers on the source graph')
    parser.add_argument('--tgt_prop', type=int, default=1, help='the number of propagation layers on the target graph')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.domain == 'Citation' or args.source == 'USA':
        args.tgt_prop = 10
    if args.domain == 'Twitch':
        args.bottleneck = 64

    # load data
    path_src = osp.join(osp.dirname(osp.realpath(__file__)), 'data/', args.domain, args.source)
    path_tgt = osp.join(osp.dirname(osp.realpath(__file__)), 'data/', args.domain, args.target)
    source_dataset, target_dataset, na_source, na_target = load_data(args, path_src, path_tgt)
    args.num_classes = len(np.unique(source_dataset[0].y.numpy()))    
    if args.domain == 'Airport':
        args.num_features = source_dataset[0].x.size(1)
    else:
        args.num_features = source_dataset.x.size(1)

    source_data = source_dataset[0].cuda()
    target_data = target_dataset[0].cuda()

    netF = BaseGNN(args).cuda()
    netC = classifier(args).cuda()
    model_pth = osp.join(osp.dirname(osp.realpath(__file__)), 'val_results', args.domain, na_source + '_' + na_target, 'model')
    modelpath = model_pth + '/model_F_' + str(0) +'.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = model_pth + '/model_C_' + str(0) +'.pt'   
    netC.load_state_dict(torch.load(modelpath))

    netF.eval()
    netC.eval()
    _, macro_f1, micro_f1, test_loss = cal_acc(target_data, netF, netC, args.tgt_prop)
    print( str('{}->{} Micro-F1:{:.2f} Macro-F1:{:.2f}').format(na_source, na_target, micro_f1 * 100, macro_f1 * 100))

