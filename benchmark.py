from models import *

import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import Planetoid

data_path = './data/Planetoid'
dataset = Planetoid(root=data_path, name='Cora')

#Get overview of dataset
print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

#Uncompress PyTorch Geomtric's representation of Adjacency Matrix
cora_adj_matrix = torch_geometric.utils.to_torch_coo_tensor(dataset.edge_index).to_dense()
cora_adj_matrix += torch.eye(dataset[0].num_nodes, dataset[0].num_nodes)
cora_adj_matrix = cora_adj_matrix.float()

print(torch.all(cora_adj_matrix.transpose(0, 1) == cora_adj_matrix)) #Check to ensure adjacency matrix is symmetric

#Get more information about dataset
print(f" The number of training instances is {torch.sum(dataset[0].train_mask)}")
print(f" The number of validation instances is {torch.sum(dataset[0].val_mask)}")
print(f" The number of test instances id {torch.sum(dataset[0].test_mask)}")

#Extract Features and perform row-wise normalization
total_data = dataset[0].x.float()
total_data = total_data/torch.linalg.norm(total_data, axis = 1, keepdims = True)

#Create Train, Val, and Test Masks as in Transductive setting
train_mask = dataset[0].train_mask
val_mask = dataset[0].val_mask
test_mask = dataset[0].test_mask
labels = dataset[0].y

num_classes = len(torch.bincount(labels))


def masked_cross_entropy(preds, labels, mask):
    preds = torch.squeeze(preds, dim = 0)
    mask_preds = preds[mask]
    mask_labels = labels[mask]
    loss = F.cross_entropy(mask_preds, mask_labels)
    return loss

def masked_accuracy(preds, labels, mask):
    preds = torch.squeeze(preds, dim = 0)
    mask_preds = preds[mask]
    mask_labels = labels[mask]
    mask_preds = torch.argmax(mask_preds, axis = 1)
    correct_predictions = (mask_preds  == mask_labels).float()
    return torch.mean(correct_predictions)

def benchmark(model):
    "Train model and return test performance"
    lr = 5e-3
    epochs = 1000
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay = 0.0005)

    best_val_loss = 0
    early_stop_counter = 0 
    patience = 100
    #best_val_loss, early_stop_counter, patience are used for early stopping purposes

    for epoch in range(0, epochs):
        model.train()
        #Forward pass
        preds = model(total_data, cora_adj_matrix)
        loss = masked_cross_entropy(preds, labels, train_mask)
        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            #Compute metrics
            model.eval()
            preds = model(total_data, cora_adj_matrix)

            train_acc = masked_accuracy(preds, labels, train_mask)
            val_acc = masked_accuracy(preds, labels, val_mask)
            val_loss = masked_cross_entropy(preds, labels, val_mask)

            print(f"Epoch: {epoch}, Training Loss: {loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            
            #Early stopping algorithm
            if epoch == 0:
                best_val_loss = val_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter +=1

        if early_stop_counter == patience:
            break

    return masked_accuracy(preds, labels, test_mask)

performances = []
for run in range(0, 2):
    gcn = NodeClassificationModel(1433, num_classes, 16, "conv")
    gcn_results = benchmark(gcn)
    performances.append((gcn_results, "conv"))


    gat = NodeClassificationModel(1433, num_classes, 8, "attn", 8)            
    gat_results = benchmark(gat)   
    performances.append((gat_results, "attn"))
 

    gat_geometric = NodeClassificationModel(1433, num_classes, 8, "geometric_attn", 8)
    gat_geometric_results = benchmark(gat_geometric)
    performances.append((gat_geometric_results, "geometric_attn"))


performances = pd.DataFrame(performances)
print(performances.groupby(1).mean())
print(performances.groupby(1).std())






   

