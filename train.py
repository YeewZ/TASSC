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

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy    

def get_neighbor_dict(data):
    in_nodes, out_nodes = data.edge_index
    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    graph_degree = len(data.edge_index[1])/len(data.x)
    return neighbor_dict, graph_degree

def get_emb(neighbor_indexes, gt_embeddings, sample_size = 5):
        sampled_embeddings = []
        if len(neighbor_indexes) < sample_size:
            sample_indexes = neighbor_indexes
            sample_indexes += np.random.choice(neighbor_indexes, sample_size - len(sample_indexes)).tolist()
        else:
            sample_indexes = random.sample(neighbor_indexes, sample_size)

        for index in sample_indexes:
            sampled_embeddings.append(gt_embeddings[index])
        return torch.stack(sampled_embeddings)

def get_emb_2(neighbor_indexes, gt_embeddings, sem_neighbor, sample_size = 5):
        sampled_embeddings = []
        sampled_embeddings.append(sem_neighbor)
        sample_size = sample_size - 1
        if len(neighbor_indexes) < sample_size:
            sample_indexes = neighbor_indexes
            sample_indexes += np.random.choice(neighbor_indexes, sample_size - len(sample_indexes)).tolist()
        else:
            sample_indexes = random.sample(neighbor_indexes, sample_size)

        for index in sample_indexes:
            sampled_embeddings.append(gt_embeddings[index])
        return torch.stack(sampled_embeddings)

def sample_neighbors(v_emd_nei, neighbor_dict, u_emd, v_emd):
    sampled_embeddings_list = []
    sampled_embeddings_neg_list = []
    for index, embedding in enumerate(u_emd):
        if index not in neighbor_dict:
            neighbor_indexes = [index]
        else:
            neighbor_indexes = neighbor_dict[index]
        sampled_embeddings = get_emb_2(neighbor_indexes, u_emd, v_emd_nei[index])
        sampled_embeddings_list.append(sampled_embeddings)
        sampled_neg_embeddings = get_emb(range(0, len(neighbor_dict)), v_emd)
        sampled_embeddings_neg_list.append(sampled_neg_embeddings)
    return torch.stack(sampled_embeddings_list), torch.stack(sampled_embeddings_neg_list)

def contrastive_loss(v_emd, v_emd_nei, neighbor_dict, tau = 1, lambda_loss=1):

    v_emd = normalize(v_emd, p=2, dim=-1)
    v_emd_nei = normalize(v_emd_nei, p=2, dim=-1)
    sampled_embeddings_u, sampled_embeddings_neg_v = sample_neighbors(v_emd_nei, neighbor_dict, v_emd, v_emd)
    projected_emd = v_emd.unsqueeze(1)

    v_emd_ = v_emd.unsqueeze(1)
    pos = torch.exp(torch.bmm(projected_emd, sampled_embeddings_u.transpose(-1, -2)).squeeze()/tau)
    neg_score = torch.log(pos + torch.sum(torch.exp(torch.bmm(v_emd_, sampled_embeddings_neg_v.transpose(-1, -2)).squeeze()/tau), dim=1).unsqueeze(-1))
    neg_score = torch.sum(neg_score, dim=1)
    pos_socre = torch.sum(torch.log(pos), dim=1)
    total_loss = torch.sum(lambda_loss * neg_score - pos_socre)
    loss = total_loss/sampled_embeddings_u.shape[0]/sampled_embeddings_u.shape[1]
    return loss


def get_semantic_neighbor(src, tgt):
    src_np = src.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()

    src_dd_ = cdist(src_np, src_np, 'cosine')
    src_dd = np.eye(src_dd_.shape[0]) * 999 + src_dd_
    src_sem_nei = np.argmin(src_dd, axis=1)

    tgt_dd_ = cdist(tgt_np, tgt_np, 'cosine')
    tgt_dd = np.eye(tgt_dd_.shape[0]) * 999 + tgt_dd_
    tgt_sem_nei = np.argmin(tgt_dd, axis=1)
    return src_sem_nei, tgt_sem_nei

def train(args, source_dataset, target_dataset, file_csv, model_path, times):

    t = time.time()
    source_data = source_dataset[0].cuda()
    target_data = target_dataset[0].cuda()

    src_neigh_dict, src_degree = get_neighbor_dict(source_data)
    tgt_neigh_dict, tgt_degree = get_neighbor_dict(target_data)
    
    ## set base network
    netF = BaseGNN(args).cuda()
    netC = classifier(args).cuda()

    # modelpath = model_pth + '/model_F_' + str(i) +'.pt'   
    # netF.load_state_dict(torch.load(modelpath))
    # modelpath = model_pth + '/model_C_' + str(i) +'.pt'   
    # netC.load_state_dict(torch.load(modelpath))

    param_group = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in netC.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)
    
    netF_dict = copy.deepcopy(netF.state_dict())
    netC_dict = copy.deepcopy(netC.state_dict())
    netF.train()
    netC.train()
    
    loss_tmp = 1e10
    for epoch in range(args.epochs):
            
        feat_src = netF(source_data.x, source_data.edge_index, args.src_prop)
        feat_tgt = netF(target_data.x, target_data.edge_index, args.tgt_prop)
        opt_src = netC(feat_src, source_data.edge_index)
        opt_tgt = netC(feat_tgt, target_data.edge_index)
        
        softmax_src = nn.Softmax(dim=1)(opt_src)
        softmax_tgt = nn.Softmax(dim=1)(opt_tgt)
        entropy_src = torch.mean(Entropy(softmax_src))
        entropy_tgt = torch.mean(Entropy(softmax_tgt))

        msoftmax_tgt = softmax_tgt.mean(dim=0)
        gentropy_tgt = torch.sum(-msoftmax_tgt * torch.log(msoftmax_tgt + args.epsilon))
        entropy_tgt -= gentropy_tgt
        
        src_sem_neighbor_idx, tgt_sem_neighbor_idx = get_semantic_neighbor(feat_src, feat_tgt)
        src_sem_nei = feat_src[src_sem_neighbor_idx]
        tgt_sem_nei = feat_tgt[tgt_sem_neighbor_idx]
        
        # ======================================= Total Loss =======================================
        train_loss = 0.5*entropy_src + F.nll_loss(F.log_softmax(opt_src, dim=1), source_data.y)
        loss = train_loss
        
        if args.domain_enhance == 1:
            loss_contrastive = contrastive_loss(feat_tgt, tgt_sem_nei, tgt_neigh_dict, tau = args.tau)
            loss = loss + loss_contrastive * args.cons_par
        elif args.domain_enhance == 3:
            loss_contrastive = contrastive_loss(feat_tgt, tgt_sem_nei, tgt_neigh_dict, tau = args.tau)
            loss_contrastive += contrastive_loss(feat_src, src_sem_nei, src_neigh_dict, tau = args.tau)
            loss = loss + loss_contrastive * args.cons_par

        loss_MMD = MMD(feat_src, feat_tgt)
        loss = loss + loss_MMD * args.weight

        if epoch > args.epochs / 2:
            loss_tos = entropy_tgt * 0.3
            loss += loss_tos
   
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        netF.eval()
        netC.eval()
        acc, _, _, _= cal_acc(source_data, netF, netC)
        _, macro_f1, micro_f1, test_loss = cal_acc(target_data, netF, netC, args.tgt_prop)
        netF.train()
        netC.train()

        if loss.data < loss_tmp:
            loss_tmp = copy.deepcopy(loss.data)
            netF_dict = copy.deepcopy(netF.state_dict())
            netC_dict = copy.deepcopy(netC.state_dict())

        print(str("" + " {0:^9}" + "|").format(epoch),
                str("" + "{:^9.6f}" + "|").format(train_loss), 
                str("" + "{:^9.6f}" + "|").format(test_loss), 
                str("" + "{:^9.2f}" + "|").format(acc * 100),
                str("" + "{:^9.2f}" + "|").format(macro_f1 * 100),
                str("" + "{:^9.2f}" + "|").format(micro_f1 * 100),
                str("" + "{:^12.6f}" + "|").format(loss))
        
        file = open(file_csv, "a+")
        print(str("" + " {0:^12}" + "|").format(epoch),
                str("" + "{:^12.6f}" + "|").format(train_loss), 
                str("" + "{:^12.6f}" + "|").format(test_loss), 
                str("" + "{:^12.2f}" + "|").format(acc * 100),
                str("" + "{:^12.2f}" + "|").format(macro_f1 * 100),
                str("" + "{:^12.2f}" + "|").format(micro_f1 * 100),
                str("" + "{:^12.6f}" + "|").format(loss),
                file=file)
        file.close()

    torch.save(netF_dict, osp.join(model_path, "model_F_" + str(times) + ".pt"))
    torch.save(netC_dict, osp.join(model_path, "model_C_" + str(times) + ".pt"))
  
    netF.load_state_dict(netF_dict)
    netC.load_state_dict(netC_dict)
    netF.eval()
    netC.eval()
    acc, _, _, _= cal_acc(source_data, netF, netC)
    _, macro_f1, micro_f1, test_loss = cal_acc(target_data, netF, netC, args.tgt_prop)
    print(str("" + " {0:^12}" + "|").format('best'),
            str("" + "{:^12.6f}" + "|").format(train_loss), 
            str("" + "{:^12.6f}" + "|").format(test_loss), 
            str("" + "{:^12.2f}" + "|").format(acc * 100),
            str("" + "{:^12.2f}" + "|").format(macro_f1 * 100),
            str("" + "{:^12.2f}" + "|").format(micro_f1 * 100))
    file = open(file_csv, "a+")
    print(str("" + " {0:^12}" + "|").format('best'),
            str("" + "{:^12.6f}" + "|").format(train_loss), 
            str("" + "{:^12.6f}" + "|").format(test_loss), 
            str("" + "{:^12.2f}" + "|").format(acc * 100),
            str("" + "{:^12.2f}" + "|").format(macro_f1 * 100),
            str("" + "{:^12.2f}" + "|").format(micro_f1 * 100),
            file=file)
    file.close()

    source_acc = copy.deepcopy(acc)
    macro_f1_tgt = copy.deepcopy(macro_f1)
    micro_f1_tgt = copy.deepcopy(micro_f1)
    test_loss = copy.deepcopy(test_loss)
    time_use = time.time() - t
    # torch.save(netF.state_dict(), osp.join(model_path, "model_F_" + str(times) + ".pt"))
    # torch.save(netC.state_dict(), osp.join(model_path, "model_C_" + str(times) + ".pt"))
    return netF, netC, time_use, source_acc, macro_f1_tgt, micro_f1_tgt, test_loss

def create_dir_par(args, need_par):
    val_index = ['Epoch', 'train_loss', 'test_loss', 'train_acc', 'Ma-F1', 'Mi-F1']
    title_fmt = str("" + " {0:^12}" + "|")
    val_index_name = ''
    for val_wrt in val_index:
        val_index_name += title_fmt.format(val_wrt)

    cur_file_name = sys.argv[0].split('/')[-1].split('.')[0]
    task_name = na_source + '_' + na_target
    save_pth = osp.join(osp.dirname(osp.realpath(__file__)), 'result_record', cur_file_name, need_par, args.domain, task_name)
    model_pth = save_pth + '/model'
    csv_pth = save_pth + '/csv'
    if not osp.exists(save_pth):
        os.system('mkdir -p ' + save_pth)
    if not osp.exists(model_pth):
        os.system('mkdir -p ' + model_pth)
    if not osp.exists(csv_pth):
        os.system('mkdir -p ' + csv_pth)
    
    return val_index_name, save_pth, model_pth, csv_pth

def create_dir(args):
    val_index = ['Epoch', 'train_loss', 'test_loss', 'train_acc', 'Ma-F1', 'Mi-F1']
    title_fmt = str("" + " {0:^12}" + "|")
    val_index_name = ''
    for val_wrt in val_index:
        val_index_name += title_fmt.format(val_wrt)

    cur_file_name = sys.argv[0].split('/')[-1].split('.')[0]
    task_name = na_source + '_' + na_target
    save_pth = osp.join(osp.dirname(osp.realpath(__file__)), 'result_record', cur_file_name, args.domain, task_name)
    model_pth = save_pth + '/model'
    csv_pth = save_pth + '/csv'
    if not osp.exists(save_pth):
        os.system('mkdir -p ' + save_pth)
    if not osp.exists(model_pth):
        os.system('mkdir -p ' + model_pth)
    if not osp.exists(csv_pth):
        os.system('mkdir -p ' + csv_pth)
    
    return val_index_name, save_pth, model_pth, csv_pth

if __name__ == '__main__':
    # model agnostic params
    parser = argparse.ArgumentParser(description='TASSC')
    parser.add_argument('--gpu', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=200, help='random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='weight decay')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers')
    parser.add_argument('--bottleneck', type=int, default=128, help='hidden size')
    parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    parser.add_argument('--times', type=int, default=5, help='run times')
    parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')

    parser.add_argument('--net', type=str, default='base', help="basemodel")
    parser.add_argument('--domain', type=str, default='Citation', help='Citation, Twitch, Airport, Blog, MAG, GraphTUDataset')
    parser.add_argument('--source', type=str, default='ACMv9', help='source domain data: DBLPv7   Citationv1   ACMv9 EUROPE USA')
    parser.add_argument('--target', type=str, default='Citationv1', help='target domain data: DBLPv7   Citationv1   ACMv9')
    parser.add_argument('--num_classes', type=int, default=None, help='class number')
    parser.add_argument('--num_features', type=int, default=None, help='original feature dimension')

    parser.add_argument('--domain_enhance', type=int, default=1, help='1: t_enhance; 2: s_enhance')
    parser.add_argument('--cons_par', type=float, default=0.5, help='constrative loss parameter')
    parser.add_argument('--tau', type=float, default=1.0, help='temp parameter')
    parser.add_argument('--weight', type=float, default=10, help='trade-off parameter')
    parser.add_argument('--src_prop', type=int, default=0, help='the number of propagation layers on the source graph')
    parser.add_argument('--tgt_prop', type=int, default=1, help='the number of propagation layers on the target graph')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # load data
    path_src = osp.join(osp.dirname(osp.realpath(__file__)), 'data/', args.domain, args.source)
    path_tgt = osp.join(osp.dirname(osp.realpath(__file__)), 'data/', args.domain, args.target)
    source_dataset, target_dataset, na_source, na_target = load_data(args, path_src, path_tgt)
    args.num_classes = len(np.unique(source_dataset[0].y.numpy()))    
    if args.domain == 'Airport':
        args.num_features = source_dataset[0].x.size(1)
    else:
        args.num_features = source_dataset.x.size(1)
    # print(print_args(args))

    lambda_val = 0.7*(source_dataset[0].x.size(0) / target_dataset[0].x.size(0)) + \
        0.3*(source_dataset[0].edge_index.size(1) / target_dataset[0].edge_index.size(1))
    
    target_degree = len(target_dataset[0].edge_index[1])/len(target_dataset[0].x)
    if lambda_val < 1.5:
        args.domain_enhance = 1
    else:
        if target_degree <= 20:
            args.domain_enhance = 3
    # print('{}->{}: {}'.format(args.source, args.target, args.domain_enhance))
    csv_index, save_pth, model_pth, csv_pth= create_dir(args)

    src_acc_dict = []
    macro_f1_dict = []
    micro_f1_dict = []
    for i in range(args.times):
        # write csv title at the first line
        print('=======================================================================================')
        file_csv = csv_pth + '/' + 'result_' + str(i) + '.csv'
        file = open(file_csv, "w+")
        print(print_args(args), file = file)
        print(csv_index, file = file)
        file.close()
        # start training...
        _, _, _, src_acc, macro_f1, micro_f1, test_loss = train(args, source_dataset, target_dataset, file_csv, model_pth, i)
        
        file = open(file_csv, "a+")
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', file=file)
        print(str("" + " {0:^17}" + ":").format('Source Accuracy'), str("" + " {:^12.2f}").format(src_acc*100),file=file)
        print(str("" + " {0:^17}" + ":").format('Macro-F1'), str("" + " {:^12.2f}").format(macro_f1*100),file=file)
        print(str("" + " {0:^17}" + ":").format('Micro-F1'), str("" + " {:^12.2f}").format(micro_f1*100),file=file)
        file.close()

        src_acc_dict.append(src_acc)
        macro_f1_dict.append(macro_f1)
        micro_f1_dict.append(micro_f1)

    mean_csv_index = ['Times', 'train_Acc', 'Macro-F1', 'Micro-F1']
    title_fmt = str("" + " {0:^12}" + "|")
    mean_csv_index_name = ''
    for val_wrt in mean_csv_index:
        mean_csv_index_name += title_fmt.format(val_wrt)
    
    file_mean_csv = save_pth + '/' + 'result_mean.csv'
    file = open(file_mean_csv, "w+")
    print(mean_csv_index_name, file = file)
    for j in range (len(macro_f1_dict)):
        print(str("" + " {0:^12}" + "|").format(j),
            str("" + "{0:^12.2f}" + "|").format(src_acc_dict[j]*100), 
            str("" + "{0:^12.2f}" + "|").format(macro_f1_dict[j]*100), 
            str("" + "{0:^12.2f}" + "|").format(micro_f1_dict[j]*100), file = file)
    print(str("" + " {0:^12}" + "|").format('mean'),
            str("" + "{0:^12.2f}" + "|").format(np.mean(src_acc_dict)*100), 
            str("" + "{0:^12.2f}" + "|").format(np.mean(macro_f1_dict)*100), 
            str("" + "{0:^12.2f}" + "|").format(np.mean(micro_f1_dict)*100), file = file)
    print(str("" + " {0:^12}" + "|").format('std'),
            str("" + "{0:^12.2f}" + "|").format(np.std(src_acc_dict)*100), 
            str("" + "{0:^12.2f}" + "|").format(np.std(macro_f1_dict)*100), 
            str("" + "{0:^12.2f}" + "|").format(np.std(micro_f1_dict)*100), file = file)
    file.close()
