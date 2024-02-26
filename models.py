""" Contains re-implementations of Graph Convolutional and Graph Attention Architectures"""
import torch
from torch import nn
import torch.nn.functional as F

from layers import *
from torch_geometric.nn import DenseGATConv
#Note: DenseGATConv is used for comparison purposes, it is NOT the implementation

class GraphConvModel(nn.Module):
    """ Implementation of KipF and Welling GCN model as per paper description"""
    def __init__(self, node_dimension, num_classes, hidden_dims):
        super().__init__()
        self.l1 = GraphConvLayer(node_dimension, hidden_dims)
        self.l2 = GraphConvLayer(hidden_dims, num_classes)
    
    def forward(self, input, adj_matrix):
        #Input Graph: N x F, where N = # of number of Nodes and F = # of features 
        #Adjacency Matrix: N x N, where elements of 1 with matrix denote the presence of edges between nodes
        
        x = F.dropout(input, p = 0.5, training = self.training)
        x = self.l1(x, adj_matrix)
        # x is now: N x hidden_dims
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)
        output = self.l2(x, adj_matrix)
        #Output: N x num_classes
        return output
    

class GraphAttentionModel(nn.Module):
    """ Implementation of Velickovic et al. GAT Architecture as per paper description"""
    def __init__(self, node_dimension, num_classes, hidden_dims, num_heads):
        super().__init__()
        self.l1 = GraphAttentionLayer(node_dimension, hidden_dims, num_heads, True)
        self.l2 = GraphAttentionLayer(hidden_dims*num_heads, num_classes, 1, False)

    def forward(self, input, adj_matrix):
        #Input Graph: N x F, where N = # of number of Nodes and F = # of features 
        #Adjacency Matrix: N x N, where elements of 1 with matrix denote the presence of edges between nodes
        x = F.dropout(input, p = 0.6, training=self.training)
        x = self.l1(x, adj_matrix)
        # x is now: N x hidden_dims*num_heads
        x = F.elu(x)
        x - F.dropout(x, p = 0.6, training = self.training)
        output = self.l2(x, adj_matrix)
        #Output: N x num_classes
        return output
    
class GraphAttentionModel_PyTorch_Geometric(nn.Module):
    """ GAT Architecture with PyTorch Geometric Layer"""
    def __init__(self, node_dimension, num_classes, hidden_dims, num_heads):
        super().__init__()
        self.l1 = DenseGATConv(node_dimension, hidden_dims, num_heads, True, dropout = 0.6)
        self.l2 = DenseGATConv(hidden_dims*num_heads, num_classes, 1, False, dropout = 0.6)
        #Note this is using PyTorch Geometric's DenseGATConv Layer

    def forward(self, input, adj_matrix):
        #Input Graph: N x F, where N = # of number of Nodes and F = # of features 
        #Adjacency Matrix: N x N, where elements of 1 with matrix denote the presence of edges between nodes
        x = F.dropout(input, p = 0.6, training=self.training)
        x = self.l1(x, adj_matrix)
        # x is now: N x hidden_dims*num_heads
        x = F.elu(x)
        x - F.dropout(x, p = 0.6, training = self.training)
        output = self.l2(x, adj_matrix)
        #Output: N x num_classes
        return output


class NodeClassificationModel(nn.Module):
    """Reusable Wrapper for Node Classification Architectures"""
    def __init__(self, node_dimension, num_classes, hidden_dims, model_choice, num_heads = None):
        super().__init__()
        self.model_choice = model_choice
        if model_choice == "attn":
            print("Using GAT")
            self.arch = GraphAttentionModel(node_dimension, num_classes, hidden_dims, num_heads)
        elif model_choice == "geometric_attn":
            print("Using PyTorch Geometric GAT")
            self.arch = GraphAttentionModel_PyTorch_Geometric(node_dimension, num_classes, hidden_dims, num_heads)
        else:
            print("Using GCN")
            self.arch = GraphConvModel(node_dimension, num_classes, hidden_dims)

    def forward(self, input, adj_matrix):
        return self.arch(input, adj_matrix)
    

