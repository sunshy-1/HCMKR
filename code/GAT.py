import math
import torch.nn.init as init

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import ones_
import manifold
from manifold.hyperboloid import Hyperboloid
from manifold.euclidean import Euclidean
from manifold.poincare import PoincareBall

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x



class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*out_features, out_features)
        self.embedding_dim=64

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.manifold = Hyperboloid()
        self.fc_hyper = HypLinear(self.manifold, 2*self.embedding_dim, self.embedding_dim, 1, 0.2, 1)


    def forward_relation(self, item_embs, entity_embs, relations, adj):
       
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1)

        hyper_a_input = self.manifold.proj_tan0(a_input, 1)
        hyper_a_input = self.manifold.expmap0(hyper_a_input, 1)
        hyper_a_input = self.manifold.proj(hyper_a_input, 1)

        hyper_relations = self.manifold.proj_tan0(relations, 1)
        hyper_relations = self.manifold.expmap0(hyper_relations, 1)
        hyper_relations = self.manifold.proj(hyper_relations, 1)

        hyper_a_input = self.fc_hyper.forward(hyper_a_input)
        hyper_dist = torch.zeros(len(hyper_a_input), len(hyper_a_input[0]))
        hyper_dist = self.manifold.sqdist(hyper_a_input.reshape(-1, self.embedding_dim), hyper_relations.reshape(-1, self.embedding_dim), 1).reshape(len(hyper_a_input),-1)
        e = 1. / (torch.exp((hyper_dist - 1) / 1) + 1.0).to(torch.device("cuda:0"))
        
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W) 
        We = torch.matmul(entity_embs, self.W)
        a_input = self._prepare_cat(Wh, We)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size()) 
        return torch.cat((Wh, We), dim=-1)


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
