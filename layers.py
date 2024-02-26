import torch
from torch import nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """ Implementation of Kipf and Welling GCN Layer as per paper description"""
    def __init__(self, in_node_dimension, out_node_dimension):
        super().__init__()
        self.W = nn.Linear(in_node_dimension, out_node_dimension)

    def forward(self, input, adj_matrix):
        #Input: NxF, where N = # of number of Nodes and F = # of features (also known as node_dimensions)
        #Adj_matrix: NxN, where elements of 1 denote the presence of edges between nodes

        norm_mtx = torch.pow(torch.sum(adj_matrix, dim = 1), -0.5)*torch.eye(len(adj_matrix), len(adj_matrix))
        #Note: Vector^-1 * Diag Matrix == inverse of (Vector*Diag Matrix)
        norm_mtx = torch.matmul(torch.matmul(norm_mtx, adj_matrix), norm_mtx)

        x = torch.matmul(norm_mtx, input)
        #Compute aggregation of features of neighbouring nodes for each node

        output = self.W(x)
        #Output: NxK, where K = out_node_dimension 
        return output


        
class GraphAttentionLayer(nn.Module):
    """ Implementation of Velickovic et al. GAT Layer as per paper description"""
    def __init__(self, in_node_dimension, out_node_dimension, num_heads, is_intermediate = True):
        super().__init__()
        self.num_heads = num_heads
        self.W = nn.Linear(in_node_dimension, out_node_dimension*num_heads)
        self.attention_weight = nn.Parameter(torch.randn(2*out_node_dimension, num_heads), requires_grad = True)
        self.is_intermediate = is_intermediate
        

    def forward(self, input, adj_matrix):
        #Input: NxF, where N = # of number of Nodes and F = # of features (also known as node_dimensions)
        #Adj_matrix: NxN, where elements of 1 denote the presence of edges between nodes
        x = self.W(input)
        #x = N x K*num_heads
        x = x.view(len(input), self.num_heads, -1)
        #x = N x num_heads x K
        x_i = torch.sum(x*(self.attention_weight[None, :int(len(self.attention_weight)/2), :].permute(0,2,1)), axis = -1)
        # self.attention_weight[None, :int(len(self.attention_weight)/2), :] dims are ... 1 x K x num_heads -> 1 x num_heads x K
        # Broadcasted multiply (N x num_heads x K, 1 x num_heads x K) -> N x num_heads x K
        # x_i -> Sum along axis -1 -> N x num_heads x K -> N x num_heads x 1
        x_j = torch.sum(x*(self.attention_weight[None, int(len(self.attention_weight)/2):, :].permute(0,2,1)), axis = -1)
        #Same as above but uses 2nd half set of weights

        sim_matrix = x_i[None, :, None] + x_j
        #(1 x N x 1 x num_heads x 1) + (N x num_heads x 1)
        #Aligning via broadcasring
        #(1 x N x 1 x num_heads x 1) 
        #(1 x 1 x N x num_heads x 1) 
        #Simplified output = Number of Nodes (batch, where each instances represents attention matrix for a single node) x Number of Nodes (where each row represents a set of attenton weights of the Node pertaining to the batch instance wrt to one of 1...N nodes) x attention heads
        sim_matrix = sim_matrix.view(len(x), len(x), -1)
        mask = torch.where(adj_matrix == 0, -float('inf'), 0)
        sim_matrix+=mask[:,:,None]
        sim_matrix = F.leaky_relu(sim_matrix)
        attention_scores = F.softmax(sim_matrix, dim=1)
        #NxNxnum_heads
        # if self.training:
        # Apply dropout to attention scores before softmax
        attention_scores = F.dropout(attention_scores, p=0.6, training=self.training)
        # attention_scores = F.dropout(attention_scores, 0.6)


        output = torch.matmul(attention_scores.permute(2,0,1) , x.permute(1,0,2))        
        if self.is_intermediate == True:
            output = torch.cat([output[i] for i in range(output.shape[0])], axis=1)
        else:
            output = torch.mean(output, axis = 0)
        return output



