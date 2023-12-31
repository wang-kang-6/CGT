import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_w(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_bool(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(False)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_rel_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


class Batch:
    def __init__(
        self,
        idx,
        attn_bias,
        attn_edge_type,
        rel_pos,
        in_degree,
        out_degree,
        x,
        edge_input,
        y,
        adj,
        w,
    ):
        super(Batch, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            attn_bias,
            attn_edge_type,
            rel_pos,
        )
        self.edge_input = edge_input
        self.adj = adj
        self.w = w

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            self.attn_bias.to(device),
            self.attn_edge_type.to(device),
            self.rel_pos.to(device),
        )
        self.edge_input = self.edge_input.to(device)
        self.adj = self.adj.to(device)
        self.w = self.w.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)


def collator(items, multi_hop_max_dist=20, rel_pos_max=20, max_node=512):

    num_virtual_tokens = 1
    
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.rel_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.adj,
            item.w,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        rel_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        adjs,
        ws,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):

        attn_biases[idx][num_virtual_tokens:, num_virtual_tokens:][
            rel_poses[idx] >= rel_pos_max
        ] = float("-inf")

    
    max_node_num = max(i.size(0) for i in xs)
    
    

    max_dist = max(i.size(-2) for i in edge_inputs)
    
    y = None
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [
            pad_attn_bias_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_biases
        ]
    )
    adj = torch.cat([pad_2d_bool(i, max_node_num + num_virtual_tokens) for i in adjs])
    
    w = torch.cat([pad_2d_w(i, max_node_num) for i in ws])

    attn_edge_type = torch.cat(
        [
            pad_edge_type_unsqueeze(i, max_node_num + num_virtual_tokens)
            for i in attn_edge_types
        ]
    )
    rel_pos = torch.cat([pad_rel_pos_unsqueeze(i, max_node_num) for i in rel_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in out_degrees])
    return Batch(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        rel_pos=rel_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        adj=adj,
        w=w,
    )


class MyGraph:
    def __init__(
        self,
        edge_index,
        x,
        edge_attr=None,
        idx=0,
        attn_bias=None,
        attn_edge_type=None,
        rel_pos=None,
        in_degree=None,
        out_degree=None,
        edge_input=None,
        y=None,
        adj=None,
        w=None,
    ):
        super(MyGraph, self).__init__()
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            attn_bias,
            attn_edge_type,
            rel_pos,
        )
        self.edge_input = edge_input
        self.adj = adj
        self.w = w

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)

        
        self.in_degree, self.out_degree = (
            self.in_degree.to(device),
            self.out_degree.to(device),
        )
        self.x = self.x.to(device)
        
        self.attn_bias, self.attn_edge_type, self.rel_pos = (
            self.attn_bias.to(device),
            self.attn_edge_type.to(device),
            self.rel_pos.to(device),
        )
        self.edge_input = self.edge_input.to(device)
        self.adj = self.adj.to(device)
        self.w = self.w.to(device)

        return self

    def __len__(self):
        return self.in_degree.size(0)