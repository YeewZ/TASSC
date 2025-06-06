import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from layers import GCNConv

class BaseGNN(nn.Module):
    def __init__(self, args, ):
        super(BaseGNN, self).__init__()
        self.relu = nn.ReLU()
        self.input_dim = args.num_features
        self.hidden_dim = args.bottleneck
        self.dropout = nn.Dropout(args.dropout_ratio)
        self.baochi = False

        self.bottlenecks = nn.ModuleList()
        self.bottlenecks.append(GCNConv(self.input_dim, self.hidden_dim, self.baochi))
        for _ in range(args.num_layers - 1):
            self.bottlenecks.append(GCNConv(self.hidden_dim, self.hidden_dim, self.baochi))
            
    def forward(self, x, edge_index, conv_time = 10):
        x1 = self.bottleneck_layers(x, edge_index, conv_time)
        return x1
    
    def bottleneck_layers(self, x, edge_index, conv_time = 10):
        for i, bottleneck in enumerate(self.bottlenecks):
            x = self.dropout(x)
            x = bottleneck(x, edge_index, conv_time)
            x = F.relu(x)
        x = self.dropout(x)
        return x

class classifier(nn.Module):
    def __init__(self, args):
        super(classifier, self).__init__()
        self.input_dim = args.bottleneck
        self.class_num = args.num_classes
        self.cls = GCNConv(self.input_dim, self.class_num)
    def forward(self, x, edge_index, conv_time = 1):
        x = self.cls(x, edge_index, conv_time)
        return x
    
