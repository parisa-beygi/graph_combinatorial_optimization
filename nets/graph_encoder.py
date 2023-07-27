import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from IPython.core.debugger import set_trace

class SkipConnectionBiInput(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input_1, input_2):
        out_1, out_2 = self.module(input_1, input_2)
        return input_1 + out_1, input_2 + out_2



class GCMultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(GCMultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        # Parameters for updating nodes' embeddins
        self.W_query_n = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key_nn = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val_nn = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_key_c = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val_c = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        # Parameters for updating colors' embeddins
        self.W_query_c = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key_n = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val_n = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out_node = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            self.W_out_color = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q_n, q_c, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        set_trace()
        # q_n = [1024, 20, 128]
        # q_c = [1024, 20, 128]
        # if h is None:
        #     h = q  # compute self-attention

        # message passing for colors
        h_c = q_n
        batch_size, graph_size, input_dim = h_c.size()
        n_query_c = q_c.size(1)

        assert q_n.size(0) == batch_size
        assert q_n.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        h_c_flat = h_c.contiguous().view(-1, input_dim)
        q_c_flat = q_c.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query_c, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q_c = torch.matmul(q_c_flat, self.W_query_c).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K_n = torch.matmul(h_c_flat, self.W_key_n).view(shp)
        V_n = torch.matmul(h_c_flat, self.W_val_n).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility_c = self.norm_factor * torch.matmul(Q_c, K_n.transpose(2, 3))

        attn_c = F.softmax(compatibility_c, dim=-1)

        heads_c = torch.matmul(attn_c, V_n)

        out_color = torch.mm(
            heads_c.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out_color.view(-1, self.embed_dim)
        ).view(batch_size, n_query_c, self.embed_dim)


        # message passing for nodes:
        # h_n = torch.cat((q_n, q_c), dim=1)
        h_nn = q_n     # self attention
        h_nc = q_c
        # h_nn and h_nc should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h_nn.size()
        n_query_n = q_n.size(1)
        assert q_n.size(0) == batch_size
        assert q_n.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # Handle messages from nodes (self attention)
        h_nn_flat = h_nn.contiguous().view(-1, input_dim)
        q_n_flat = q_n.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        # shp = (8, 1024, 20, -1)
        # shp_q = (8, 1024, 20, -1)
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query_n, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        # Q = [8, 1024, 20, 16]
        Q_n = torch.matmul(q_n_flat, self.W_query_n).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        # K = [8, 1024, 20, 16]
        K_nn = torch.matmul(h_nn_flat, self.W_key_nn).view(shp)
        # V = [8, 1024, 20, 16]
        V_nn = torch.matmul(h_nn_flat, self.W_val_nn).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        # compatibility = [8, 1024, 20, 20]
        compatibility_nn = self.norm_factor * torch.matmul(Q_n, K_nn.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query_n, graph_size).expand_as(compatibility_nn)
            compatibility_nn[mask] = -np.inf

        attn_nn = F.softmax(compatibility_nn, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc_nn = attn_nn.clone()
            attnc_nn[mask] = 0
            attn_nn = attnc_nn

        heads_nn = torch.matmul(attn_nn, V_nn)
        # heads.shape = [8, 1024, 20, 16]

        # Handle messages from colors
        h_nc_flat = h_nc.contiguous().view(-1, input_dim)
        K_nc = torch.matmul(h_nc_flat, self.W_key_c).view(shp)
        V_nc = torch.matmul(h_nc_flat, self.W_val_c).view(shp)
        compatibility_nc = self.norm_factor * torch.matmul(Q_n, K_nc.transpose(2, 3))

        attn_nc = F.softmax(compatibility_nc, dim=-1)

        heads_nc = torch.matmul(attn_nc, V_nc)
        # heads.shape = [8, 1024, 20, 16]

        heads = torch.add(heads_nn, heads_nc)
        # set_trace()
        # self.W_out.shape  = ([8, 16, 128])
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query_n, self.embed_dim)

        return out, out_color


class GCMultiHeadAttentionLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(GCMultiHeadAttentionLayer, self).__init__()
        self.skip_multihead = SkipConnectionBiInput(GCMultiHeadAttention(n_heads, input_dim=embed_dim, embed_dim=embed_dim))
        self.norm_multihead_n = Normalization(embed_dim, normalization)
        self.norm_multihead_c = Normalization(embed_dim, normalization)
        self.skip_ff_n = SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        )
        self.skip_ff_c = SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        )

        self.norm_ff_n = Normalization(embed_dim, normalization)
        self.norm_ff_c = Normalization(embed_dim, normalization)

    def forward(self, x_n, x_c):

        x_n, x_c = self.skip_multihead(x_n, x_c)
        x_n = self.norm_multihead_n(x_n)
        x_c = self.norm_multihead_c(x_c)

        x_n = self.skip_ff_n(x_n)
        x_c = self.skip_ff_c(x_c)

        return (self.norm_ff_n(x_n), self.norm_ff_c(x_c))


class GCGraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GCGraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.ModuleList([GCMultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)]
        )

    def forward(self, x_n, x_c, mask=None):
        # set_trace()

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h_n = self.init_embed(x_n.view(-1, x_n.size(-1))).view(*x_n.size()[:2], -1) if self.init_embed is not None else x_n
        h_c = self.init_embed(x_c.view(-1, x_c.size(-1))).view(*x_c.size()[:2], -1) if self.init_embed is not None else x_c

        h_n, h_c = self.layers(h_n, h_c)

        return (
            h_n,  # nodes: (batch_size, graph_size, embed_dim)
            h_c,  # colors: (batch_size, graph_size, embed_dim)
            h_n.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        set_trace()
        # q = [1024, 20, 128]
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # hflat.shape = [20480, 128]
        # qflat.shape = [20480, 128]
        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        # shp = (8, 1024, 20, -1)
        # shp_q = (8, 1024, 20, -1)
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        # Q = [8, 1024, 20, 16]
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        # K = [8, 1024, 20, 16]
        K = torch.matmul(hflat, self.W_key).view(shp)
        # V = [8, 1024, 20, 16]
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        # compatibility = [8, 1024, 20, 20]
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        # set_trace()
        # self.W_out.shape  = ([8, 16, 128])
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        # set_trace()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        # set_trace()

        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

