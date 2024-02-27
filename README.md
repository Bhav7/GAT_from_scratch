# Graph Neural Networks From Scratch

![teaser](Velickovic_et_al-1.jpg)

This repository includes code which aims to reimplement the Graph Convolutional (GCN) Architecture from KipF and Welling's paper: [Semi-Supervised Classification with Graph Convolutional Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjH3ICgqsqEAxVE7skDHSe1B0cQFnoECAYQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1609.02907&usg=AOvVaw1HSQRqpg9PIWjueBnAIuC8&opi=89978449) and the Graph Attention Architecture (GAT) from Velickovic et al's paper: [Graph Attention Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj7gZ_PqsqEAxVL7skDHYJgCE4QFnoECBEQAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1710.10903&usg=AOvVaw3V0c3RJ86MZ70WLK2qpipV&opi=89978449) using only PyTorch 

To evaluate the implementations, GCNs and GATs were trained and tested on the CORA citation network dataset for the purpose of node classification. Additionally, a PyTorch Geometric variant of a GAT were compoared against from the from-scratch GNNs.

To run the code (assuming dependencies are installed: see imported packages in benchmark.py):

```
$ python benchmark.py 
```
Average test results over 5 runs:

```
GCN: 0.808
GAT: 0.815
PyG GAT: 0.822
```