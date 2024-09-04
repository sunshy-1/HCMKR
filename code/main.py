import world
import utils
import torch
import numpy as np
import time
import Procedure
import manifold
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from tensorboardX import SummaryWriter
from os.path import join
from torch_sparse.tensor import to
from tqdm import tqdm
from contrast import Contrast
from torch import optim
from torch.optim import optimizer, lr_scheduler
from world import cprint

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================

import register
from register import dataset, kg_dataset
Recmodel = register.MODELS[world.model_name](world.config, dataset, kg_dataset)
Recmodel = Recmodel.to(world.device)
contrast_model = Contrast(Recmodel).to(world.device)
optimizer = optim.Adam(Recmodel.parameters(), world.config['lr'])

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma = 0.2)

bpr = utils.BPRLoss(Recmodel, optimizer)
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")

if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment))
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    least_loss = 1e5
    best_result = 0.
    stopping_step = 0
    
    for epoch in tqdm(range(world.TRAIN_epochs), disable=True):
        start = time.time()
        # transR learning
        if epoch%1 == 0:
            if world.train_trans:
                cprint("[Trans]")
                trans_loss = Procedure.TransR_train(Recmodel, optimizer)
                print(f"trans Loss: {trans_loss:.3f}")

        # joint learning part
        if not world.pretrain_kgc:
            if world.kgc_joint:
                contrast_views = contrast_model.get_views()
            else:
                contrast_views = contrast_model.get_views("ui")
            cprint("[Joint Learning]")
            if world.kgc_joint or world.uicontrast!="NO":
                output_information, all_user_emb = Procedure.BPR_train_contrast(dataset, Recmodel, bpr, contrast_model, contrast_views, epoch, optimizer, neg_k=Neg_k,w=w)
            else:
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            
            endl = time.time()
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}', endl-start)
            if epoch<world.test_start_epoch:
                if epoch %1 == 0:
                    cprint("[TEST]")
                    Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            else:
                if epoch % world.test_verbose == 0:
                    cprint("[TEST]")
                    result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                    if result["recall"] > best_result:
                        stopping_step = 0
                        best_result = result["recall"]
                        torch.save(Recmodel.state_dict(), weight_file)
                        if world.save_emb:
                            best_user_emb = all_user_emb
                            import pickle
                            pic = open('useremb-nohyper-'+str(world.dataset)+'.pkl', 'wb')
                            pickle.dump(best_user_emb.cpu(),pic)
                            pic.close()

                    else:
                        stopping_step += 1
                        if stopping_step >= world.early_stop_cnt:
                            print(f"early stop triggerd at epoch {epoch}")
                            break
        
        scheduler.step()
finally:
    if world.tensorboard:
        w.close()