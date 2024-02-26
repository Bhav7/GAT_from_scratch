""" Contains re-implementations of Graph Convolutional and Graph Attention Layers """
import torch
from torch import nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """ Implementation of Kipf and Welling GCN Layer as per paper description"""
    def __init__(self, in_node_dimension, out_node_dimension):
        super().__init__()
        self.W = nn.Linear(in_node_dimension, out_node_dimension)

    def forward(self, input, adj_matrix):
        #Input Graph: N x F, where N = # of number of Nodes and F = # of features 
        #Adjacency Matrix: N x N, where elements of 1 with matrix denote the presence of edges between nodes

        norm_mtx = torch.pow(torch.sum(adj_matrix, dim = 1), -0.5)*torch.eye(len(adj_matrix), len(adj_matrix)) #Note: Vector^-1 * Diag Matrix == inverse of (Vector*Diag Matrix)
        norm_mtx = torch.matmul(torch.matmul(norm_mtx, adj_matrix), norm_mtx)
        x = torch.matmul(norm_mtx, input)
        output = self.W(x)
        #Output: N x K, where K = out_node_dimension 
        return output


        
class GraphAttentionLayer(nn.Module):
    """ Implementation of Velickovic et al. GAT Layer as per paper description"""
    def __init__(self, in_node_dimension, out_node_dimension, num_heads, is_intermediate = True):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.Linear(in_node_dimension, out_node_dimension*num_heads)
        self.attention_weight = nn.Parameter(torch.randn(2*out_node_dimension, num_heads), requires_grad = True)
        self.is_intermediate = is_intermediate #Flag needed to determine if GAT Layer generates final outputs or intermediate values for subsequent GAT Layer
        

    def forward(self, input, adj_matrix):
        #Input Graph: N x F, where N = # of number of Nodes and F = # of features 
        #Adjacency Matrix: N x N, where elements of 1 with matrix denote the presence of edges between nodes

        x = self.W(input)
        #x is now: N x K*num_heads, where K = out_node_dimension
        x = x.view(len(input), self.num_heads, -1)
        #x is now: N x num_heads x K

        x_i = torch.sum(x*(self.attention_weight[None, :int(len(self.attention_weight)/2), :].permute(0,2,1)), axis = -1)
        #Breakdown ...
        #   1) self.attention_weight[None, :int(len(self.attention_weight)/2), :] shape is : 1 x K x num_heads
        #   2) self.attention_weight[None, :int(len(self.attention_weight)/2), :].permute(0,2,1) shape is: 1 x num_heads x K
        #   3) x*(self.attention_weight[None, :int(len(self.attention_weight)/2), :].permute(0,2,1) shape is: N x num_heads x K
        #   4) x_i shape is: N x num_heads x 1


        x_j = torch.sum(x*(self.attention_weight[None, int(len(self.attention_weight)/2):, :].permute(0,2,1)), axis = -1)
        #x_j shape is: N x num_heads x 1

        sim_matrix = x_i[None, :, None] + x_j
        #Breakdown ...
        #   1) x_i[None, :, None] shape is: 1 x N x 1 x num_heads x 1
        #   2) To add x_i to x_j, broadcasting semantics treat x_j's shape as: (1 x 1 x N x num_heads x 1)
        #   3) sim_matrix shape is: 1 x N x N x num_heads x 1

        sim_matrix = sim_matrix.view(len(x), len(x), -1)
        #sim_matrix shape is: N x N x num_heads 
        mask = torch.where(adj_matrix == 0, -float('inf'), 0)
        sim_matrix+=mask[:,:,None]
        sim_matrix = F.leaky_relu(sim_matrix)
        attention_scores = F.softmax(sim_matrix, dim=1)
        #attention_scores shape is N x N x num_heads

        attention_scores = F.dropout(attention_scores, p=0.6, training=self.training) #Ensures dropout applied only during training

        output = torch.matmul(attention_scores.permute(2,0,1) , x.permute(1,0,2)) 
        #Output: num_heads x N x K
        if self.is_intermediate == True:
            output = torch.cat([output[i] for i in range(output.shape[0])], axis=1) #Note: Would have preferred torch.view, but order gets jumbled up
            #Output: N x K*num_heads
        else:
            output = torch.mean(output, axis = 0)
            #Output: N x K
        return output



