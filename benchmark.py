from models import *

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torch_geometric
from torch_geometric.datasets import Planetoid

data_path = './data/Planetoid'
dataset = Planetoid(root=data_path, name='Cora')
print(f'Dataset: {dataset}:')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

cora_adj_matrix = torch_geometric.utils.to_torch_coo_tensor(dataset.edge_index).to_dense()
cora_adj_matrix += torch.eye(dataset[0].num_nodes, dataset[0].num_nodes)
cora_adj_matrix = cora_adj_matrix.float()
#Check to ensure mtx is symmetric
print(torch.all(cora_adj_matrix.transpose(0, 1) == cora_adj_matrix))

print(f" The number of training instances is {torch.sum(dataset[0].train_mask)}")
print(f" The number of validation instances is {torch.sum(dataset[0].val_mask)}")
print(f" The number of test instances id {torch.sum(dataset[0].test_mask)}")


total_data = dataset[0].x.float()
#Row-wise normalization
total_data = total_data/torch.linalg.norm(total_data, axis = 1, keepdims = True)


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
    # lr = 5e-3
    lr = 0.01
    epochs = 25
    optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay = 0.0005)
    best_val_loss = 0
    early_stop_counter = 0
    for epoch in range(0, epochs):
        model.train()
        preds = model(total_data, cora_adj_matrix)
        loss = masked_cross_entropy(preds, labels, train_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            preds = model(total_data, cora_adj_matrix)

            train_acc = masked_accuracy(preds, labels, train_mask)
            val_acc = masked_accuracy(preds, labels, val_mask)
            val_loss = masked_cross_entropy(preds, labels, val_mask)

            print(f"Epoch: {epoch}, Training Loss: {loss}, Training Accuracy: {train_acc}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")
            
            if epoch == 0:
                best_val_loss = val_loss
            
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter +=1

        if early_stop_counter == 100:
            break

    return masked_accuracy(preds, labels, test_mask)

gcn = NodeClassificationModel(1433, num_classes, 16, "conv")
gcn_results =benchmark(gcn)
print(f"The test set performance using the Graph Convolutional NN is {gcn_results}")

gat = NodeClassificationModel(1433, num_classes, 8, "attn", 8)            
gat_results = benchmark(gat)    
print(f"The test set performance using the Graph Attention NN is {gat_results}")

gat_geometric = NodeClassificationModel(1433, num_classes, 8, "geometric_attn", 8)
gat_geometric_results = benchmark(gat_geometric)
print(f"The test set performance using Pytorch Geometric Graph Attention NN is {gat_geometric_results}")







   

