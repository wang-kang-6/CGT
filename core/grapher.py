import dgl
import dgl.function as fn
from typing import Dict, Union, Tuple
import torch
import torch.nn as nn
import os
import math
import numpy as np

from histocartography.ml.layers.mlp import MLP
from histocartography.ml.models.base_model import BaseModel
from histocartography.ml import MultiLayerGNN
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN
from histocartography.ml.models.zoo import MODEL_NAME_TO_URL, MODEL_NAME_TO_CONFIG
from histocartography.utils import download_box_link
from collator import *
from wrapper import preprocess_item
from train_utils import parse_device
from dataloader import set_graph_on_cuda

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

DEVICE = parse_device()

def init_bert_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class Grapher(BaseModel):
    """
    CGT Model.
    """

    def __init__(
            self,
            gnn_params: Dict,
            classification_params: Dict,
            node_dim: int,
            n_layers,
            num_heads,
            hidden_dim,
            dropout_rate,
            intput_dropout_rate,
            weight_decay,
            ffn_dim,
            dataset_name,
            warmup_updates,
            tot_updates,
            peak_lr,
            end_lr,
            edge_type,
            multi_hop_max_dist,
            attention_dropout_rate,
            flag=False,
            flag_m=3,
            flag_step_size=1e-3,
            flag_mag=1e-3,
            **kwargs,
    ):
        """
        CGT constructor.

        Args:
            gnn_params (Dict): GNN configuration parameters.
            classification_params (Dict): classification configuration parameters.
            node_dim (int): Tissue node feature dimension.
        """

        super().__init__(**kwargs)
        
        
        self.node_dim = node_dim
        num_virtual_tokens = 1

        self.num_heads = num_heads
        self.w_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(1024, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [
            EncoderLayer(
                hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.downstream_out_proj = nn.Linear(
            hidden_dim, 7)

        self.graph_token = nn.Embedding(num_virtual_tokens, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(num_virtual_tokens, num_heads)

        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_bert_params(module, n_layers=n_layers))

    def forward(
        self,
        graph: Union[dgl.DGLGraph, Tuple[torch.tensor, torch.tensor]],
        ws: torch.Tensor,
    ) -> torch.tensor:
        """
        Foward pass.

        Args:
            graph (Union[dgl.DGLGraph, Tuple[torch.tensor, torch.tensor]]): Tissue graph to process.

        Returns:
            torch.tensor: Model output.
        """

        '''
        Unbatch graphs
        '''
        num_virtual_tokens = 1
        graphs = dgl.unbatch(graph)
        graph_embeddings, in_degrees, attn_biases = [], [], []
        items = []
        for i in range(len(graphs)):
            g = graphs[i]
            x = g.ndata[GNN_NODE_FEAT_IN]
            e = g.edges()
            e = torch.cat((e[0].unsqueeze(0), e[1].unsqueeze(0)), 0)
            x = x.detach().cpu()
            item = MyGraph(edge_index=e, x=x, w=ws[i])
            item = preprocess_item(item)
            items.append(item.to(DEVICE))


        '''
            batched_data
        '''
        batched_data = collator(items)
        del items
        attn_bias, rel_pos, x = (
            batched_data.attn_bias,
            batched_data.rel_pos,
            batched_data.x,
        )
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree

        edge_input, attn_edge_type = (
            batched_data.edge_input,
            batched_data.attn_edge_type,
        )
        adj = batched_data.adj
        w = batched_data.w

        '''
            graph_attn_bias
        '''
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        )  

        w_bias = self.w_encoder(w).permute(0, 3, 1, 2)
        
        rel_pos_bias = w_bias
        graph_attn_bias[:, :, num_virtual_tokens:, num_virtual_tokens:] = (
                graph_attn_bias[:, :, num_virtual_tokens:, num_virtual_tokens:]
                + rel_pos_bias
        )

        t = self.graph_token_virtual_distance.weight.view(
            1, self.num_heads, num_virtual_tokens
        ).unsqueeze(
            -2
        )  
        self.graph_token_virtual_distance.weight.view(
            1, self.num_heads, num_virtual_tokens
        ).unsqueeze(-2)
        graph_attn_bias[:, :, num_virtual_tokens:, :num_virtual_tokens] = (
                graph_attn_bias[:, :, num_virtual_tokens:, :num_virtual_tokens]
                + t  
        )

        '''
        node feature + graph token
        '''
        node_feature = x
        p1 = self.in_degree_encoder(in_degree)
        node_feature = (
                node_feature
                + 2*p1
        )

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        p_feature = torch.cat([graph_token_feature, p1], dim=1)

        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            for i in range(output.size()[0]):
                adj_matrix = adj[i, :, :].detach().cpu().numpy()
                adj_matrix[-1, :] = True
                adj_matrix[:, -1] = True

                src, dst = np.nonzero(adj_matrix)
                sg = dgl.graph((src, dst))
                
                sg.ndata['feat'] = p_feature[i, :, :].detach().cpu()
                f1 = sg.ndata['feat']
                sg.update_all(fn.copy_u('feat', 'message'), fn.sum('message', 'feat'))
                f2 = sg.ndata['feat'].to(DEVICE)

                output[i, :, :] = output[i, :, :] + f2

            output = enc_layer(output, graph_attn_bias, mask=None, w=None)  
        output = self.final_ln(output)

        output = self.downstream_out_proj(output[:, 0, :])

        return output

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument("--n_layers", type=int, default=6)
        parser.add_argument("--num_heads", type=int, default=6)
        parser.add_argument("--hidden_dim", type=int, default=514)
        parser.add_argument("--ffn_dim", type=int, default=256)

        parser.add_argument("--intput_dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--attention_dropout_rate", type=float, default=0.1)
        parser.add_argument("--checkpoint_path", type=str, default="")
        parser.add_argument("--warmup_updates", type=int, default=60000)
        parser.add_argument("--tot_updates", type=int, default=1000000)
        parser.add_argument("--peak_lr", type=float, default=2e-4)
        parser.add_argument("--end_lr", type=float, default=1e-9)
        parser.add_argument("--edge_type", type=str, default="multi_hop")
        parser.add_argument("--validate", action="store_true", default=False)
        parser.add_argument("--test", action="store_true", default=False)
        parser.add_argument("--flag", action="store_true")
        parser.add_argument("--flag_m", type=int, default=3)
        parser.add_argument("--flag_step_size", type=float, default=1e-3)
        parser.add_argument("--flag_mag", type=float, default=1e-3)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None, mask=None, w=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  
        v = v.transpose(1, 2)  
        k = k.transpose(1, 2).transpose(2, 3)  


        q = q * self.scale
        x = torch.matmul(q, k)  
        if attn_bias is not None:
            x = x + attn_bias
        if mask is not None:
            mask = mask.unsqueeze(1)
            x = x.masked_fill(mask, 0)
        if w is not None:
            w = w.unsqueeze(1)
            w = w.repeat(1, self.num_heads, 1, 1)
            x = x.matmul(w)

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  

        x = x.transpose(1, 2).contiguous()  
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):
    def __init__(
            self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads
    ):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, mask=None, w=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias, mask=mask, w=w)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


