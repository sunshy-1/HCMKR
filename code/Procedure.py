import world
import numpy as np
import torch
import utils
import dataloader
import model
import multiprocessing

from contrast import Contrast
from torch.utils.data.dataloader import DataLoader
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.metrics import roc_auc_score
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm

CORES = multiprocessing.cpu_count() // 2

def TransR_train(recommend_model, opt):
    Recmodel = recommend_model
    Recmodel.train()
    kgdataset = dataloader.KGDataset()
    kgloader = DataLoader(kgdataset,batch_size=4096,drop_last=True)
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(world.device)
        relations = data[1].to(world.device)
        pos_tails = data[2].to(world.device)
        neg_tails = data[3].to(world.device)
        kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()


def train_contrast(recommend_model, contrast_model, contrast_views, optimizer):
    recmodel = recommend_model
    recmodel.train()
    aver_loss = 0.

    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]

    l_kg = list()
    l_item = list()
    l_user = list()

    if world.kgc_enable:
        kgv1_readouts = recmodel.cal_item_embedding_from_kg(kgv1).split(2048)
        kgv2_readouts = recmodel.cal_item_embedding_from_kg(kgv2).split(2048)

        for kgv1_ro, kgv2_ro in zip(kgv1_readouts, kgv2_readouts):
            l_kg.append(contrast_model.semi_loss(kgv1_ro, kgv2_ro).sum())

    l_contrast = torch.stack(l_kg).sum()
    optimizer.zero_grad()
    l_contrast.backward()
    optimizer.step()

    aver_loss += l_contrast.cpu().item() / len(kgv1_readouts)
    return aver_loss

def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model :Contrast, contrast_views, epoch, optimizer, neg_k=1, w=None):
    Recmodel :model.HCMKR = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    batch_size = world.config['bpr_batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)

    total_batch = len(dataloader)
    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.
    
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"]
    
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader),disable=True):
        batch_users = train_data[0].long().to(world.device)
        batch_pos = train_data[1].long().to(world.device)
        batch_neg = train_data[2].long().to(world.device)

        l_main = bpr.compute(batch_users, batch_pos, batch_neg)
        l_ssl = list()
        items = batch_pos

        if world.uicontrast!="NO":
            if world.kgc_joint:
                '''
                view_computer_all_hyper(uiv1, kgv1, prune = 0 or 1, skip_layer = 'before' or 'after')
                prune = 1, skip_layer = None: enable prune contrastive
                prune = 0, skip_layer = 'before' or 'after': enable skip-layer contrastive
                '''
                if world.contrast_level == 'drop':
                    usersv1_ro, itemsv1_ro = Recmodel.view_computer_all_hyper(uiv1, kgv1, 0, 'none')
                    usersv2_ro, itemsv2_ro = Recmodel.view_computer_all_hyper(uiv2, kgv2, 0, 'none')
                elif world.contrast_level == 'cross':
                    usersv1_ro, itemsv1_ro = Recmodel.view_computer_all_hyper(uiv1, kgv1, 0, 'before')
                    usersv2_ro, itemsv2_ro = Recmodel.view_computer_all_hyper(uiv2, kgv2, 0, 'after')
                elif world.contrast_level == 'prune':
                    usersv1_ro, itemsv1_ro = Recmodel.view_computer_all_hyper(uiv1, kgv1, 1, 'none')
                    usersv2_ro, itemsv2_ro = Recmodel.view_computer_all_hyper(uiv2, kgv2, 0, 'none')
                else:
                    raise NotImplementedError("This constrast function has not been implemented yet!") 

            else:
                usersv1_ro, itemsv1_ro = Recmodel.view_computer_ui(uiv1)
                usersv2_ro, itemsv2_ro = Recmodel.view_computer_ui(uiv2)

            items_uiv1 = itemsv1_ro[items]
            items_uiv2 = itemsv2_ro[items]
            l_item = contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro, items)

            users = batch_users
            users_uiv1 = usersv1_ro[users]
            users_uiv2 = usersv2_ro[users]
            l_user = contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro, users)
            
            l_ssl.extend([l_user*world.ssl_reg, l_item*world.ssl_reg])
            
        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main + l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()

        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', l_all, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}", usersv1_ro


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Main"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.HCMKR

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []

        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))

        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}',
                          {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                          {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                          {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        if multicore == 1:
            pool.close()
        print(results)
        
        with open('Logs.txt','a') as file:
            file.write(str(results)+'\n')
        return results
