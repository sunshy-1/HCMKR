import os
import json
from os.path import join
import torch
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


all_dataset = ['yelp2018', 'amazon-book', 'ml-20m']
all_models  = ['lgn','hcmkr','sgl']

with open('./configs/config.json', 'r') as f:
    all_configs = json.load(f)
config = all_configs[args.contrast_level+'_'+args.dataset]

GPU = torch.cuda.is_available()
device = torch.device("cuda:0")
num_clusters = args.num_clusters
early_stop_cnt = args.early_stop_cnt

kgcn = "RGAT"
train_trans = False
entity_num_per_item = 10

uicontrast = "RANDOM"
kgc_enable = True
kgc_joint = True
kgc_temp = 0.2
use_kgc_pretrain = False
pretrain_kgc = False
kg_p_drop = 0.5
ui_p_drop = 0.001
ssl_reg = 0.1
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

test_verbose = 1
test_start_epoch = 1

dataset = args.dataset
config['dataset'] = args.dataset
contrast_level = args.contrast_level
save_emb = args.save_emb
ssl_reg = config['ssl_reg_set']

if dataset=='amazon-book':
    config['dropout'] = 1
    config['keep_prob']  = 0.8
    uicontrast = "YES"
    ui_p_drop = 0.05
    mix_ratio = 0.75
    test_start_epoch = 15

elif dataset == 'ml-20m':
    config['lr'] = 1e-3
    config['dropout'] = 1
    config['keep_prob'] = 0.8
    uicontrast = "YES"
    ui_p_drop = 0.05
    mix_ratio = 0.75
    test_start_epoch = 1

elif dataset=='yelp2018':
    config['dropout'] = 1
    config['keep_prob']  = 0.8
    uicontrast = "YES"
    ui_p_drop = 0.1
    test_start_epoch = 25


model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")

if model_name == 'lgn':
    kgcn = "NO"
    train_trans = False
    uicontrast = "NO"
    kgc_enable = False
    kgc_joint = False
    use_kgc_pretrain = False
    pretrain_kgc = False
elif model_name == 'sgl':
    kgcn = "NO"
    train_trans = False
    uicontrast = "RANDOM"
    kgc_enable = False
    kgc_joint = False
    use_kgc_pretrain = False
    pretrain_kgc = False

TRAIN_epochs = config['epochs']
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)

def cprint(words : str):
    print(words)
