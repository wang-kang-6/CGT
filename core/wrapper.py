import random
import torch
import numpy as np
import torch_geometric.datasets


import pyximport
pyximport.install()

pyximport.install(setup_args={"include_dirs": np.get_include()})
import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    feature_offset = feature_offset
    x = x + feature_offset
    return x


def preprocess_item(item):

    num_virtual_tokens = 1
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x

    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    N = x.size(0)


    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True


    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )  

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    
    

    
    

    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    

    
    item.edge_attr = edge_attr
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()

    

    return item



