import os
import world
import torch
from dataloader import BasicDataset
from torch import nn
from GAT import GAT
import numpy as np
from utils import _L2_loss_mean
import torch.nn.functional as F
import time
import manifold
import torch.nn.utils.prune as prune_torch
from manifold.hyperboloid import Hyperboloid
from manifold.euclidean import Euclidean
from manifold.poincare import PoincareBall


import layers.hyp_layers as hyp_layers

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    

class HCMKR(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset,
                 kg_dataset):
        super(HCMKR, self).__init__()
        self.config = config
        self.dataset : BasicDataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.level_dict = {}

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)

        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config['pretrain'] == 0:
            world.cprint('use NORMAL distribution UI')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution ENTITY')
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        print(f"HCMKR is ready to go!")

        self.c = nn.Parameter(torch.tensor([1.0]*(self.n_layers+1), dtype=torch.float), requires_grad=True).to(world.device)
        self.manifold = Hyperboloid()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.0 if world.contrast_level=='prune' else 0.2, alpha=0.2).train()
        self.prune_gat = GAT(self.latent_dim, self.latent_dim, dropout=0.0 if world.contrast_level=='prune' else 0.2, alpha=0.2).train()
        
        hgc_layers = []
        
        for i in range(self.n_layers):
            c_in, c_out = self.c[i], self.c[i+1]
            in_dim, out_dim = 64, 64
            act = getattr(F, 'relu')
            hgc_layers.append(
                    hyp_layers.HyperbolicGraphConvolution(
                            self.manifold, in_dim, out_dim, c_in, c_out, 0.0 if world.contrast_level=='prune' else 0.2, act, 1, 0, 0
                    ).to(world.device)
            )
        self.layers = hgc_layers

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def view_computer_all_hyper(self, g_droped, kg_droped, prune=None, level=None):
        """
        propagate methods for contrastive hyperbolic gcn
        """   
        if level == 'after':
            users, items = torch.split(self.level_dict['after'], [self.num_users, self.num_items])
            return users, items

        users_emb = self.embedding_user.weight
        if prune:
            items_emb = self.cal_item_embedding_from_kg(kg_droped,1)
        else:
            items_emb = self.cal_item_embedding_from_kg(kg_droped,0)
        all_emb = torch.cat([users_emb, items_emb])
        
        all_emb_tan = self.manifold.proj_tan0(all_emb, self.c[0])
        all_emb_hyp = self.manifold.expmap0(all_emb_tan, c=self.c[0])
        all_emb_hyp = self.manifold.proj(all_emb_hyp, c=self.c[0])

        embs_hyper = [all_emb_hyp]
        for i in range(self.n_layers):
            input_hgcn = (F.dropout(embs_hyper[i], 0.0 if world.contrast_level=='prune' else 0.2), g_droped)
            emb_hyper,_ = self.layers[i].forward(input_hgcn)
            embs_hyper.append(emb_hyper)

        for i in range(len(embs_hyper)):
            embs_hyper[i] = self.manifold.logmap0(embs_hyper[i], self.c[i])
        users, items = torch.split(torch.mean(torch.stack(embs_hyper, dim=1),dim=1), [self.num_users, self.num_items])
        
        if level == 'before':
            contrast_layer_num = -3
            self.level_dict['before'] = embs_hyper[contrast_layer_num]
            self.level_dict['after'] = torch.mean(torch.stack(embs_hyper, dim=1), dim=1)
            users, items = torch.split(self.level_dict['before'], [self.num_users, self.num_items])
        
        return users, items



    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def computer_hyper(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        all_emb_tan = self.manifold.proj_tan0(all_emb, self.c[0])
        all_emb_hyp = self.manifold.expmap0(all_emb_tan, c=self.c[0])
        all_emb_hyp = self.manifold.proj(all_emb_hyp, c=self.c[0])

        embs_hyper = [all_emb_hyp]
        for i in range(self.n_layers):
            input_hgcn = (embs_hyper[i], g_droped)
            emb_hyper,_ = self.layers[i].forward(input_hgcn)
            embs_hyper.append(emb_hyper)

        for i in range(len(embs_hyper)):
            embs_hyper[i] = self.manifold.logmap0(embs_hyper[i], self.c[i])
        users, items = torch.split(torch.mean(torch.stack(embs_hyper, dim=1),dim=1), [self.num_users, self.num_items])

        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer_hyper()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))

        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        return loss, reg_loss
       
       
    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)                
        h_embed = self.embedding_item(h)              
        pos_t_embed = self.embedding_entity(pos_t)      
        neg_t_embed = self.embedding_entity(neg_t)     

        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + 1e-3 * l2_loss
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)                 
        W_r = self.W_R[r]                                

        h_embed = self.embedding_item(h)              
        pos_t_embed = self.embedding_entity(pos_t)      
        neg_t_embed = self.embedding_entity(neg_t)      

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     

        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     

        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + 1e-3 * l2_loss

        return loss

    def cal_item_embedding_gat(self, kg:dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(world.device)) #item_num, emb_dim
        item_entities = torch.stack(list(kg.values())) 
        entity_embs = self.embedding_entity(item_entities) 
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        
        return self.gat(item_embs, entity_embs, padding_mask)
    
    def cal_item_embedding_rgat(self, kg:dict, prune=None):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(world.device))
        item_entities = torch.stack(list(kg.values()))
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities)
        relation_embs = self.embedding_relation(item_relations) 
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        
        if prune:
            for (adv_name,adv_param), (name,param) in zip(self.prune_gat.named_parameters(), self.gat.named_parameters()):
                adv_param.data = param.data
            
            prune_module = self.prune_gat.layer.fc_hyper
            prune_torch.random_unstructured(prune_module, name="weight", amount=0.3)
            return self.prune_gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)

        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)


    def cal_item_embedding_from_kg(self, kg: dict, prune=None):
        if kg is None:
            kg = self.kg_dict
        
        if(world.kgcn=="GAT"):
            return self.cal_item_embedding_gat(kg)
        elif world.kgcn=="RGAT":
            return self.cal_item_embedding_rgat(kg, prune)
        elif(world.kgcn=="MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif(world.kgcn=="NO"):
            return self.embedding_item.weight


    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(world.device)) #item_num, emb_dim
        item_entities = torch.stack(list(kg.values()))
        entity_embs = self.embedding_entity(item_entities) 
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        
        return item_embs+entity_embs_mean


    def forward(self, users, items):
        all_users, all_items = self.computer_hyper()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        
        return gamma
