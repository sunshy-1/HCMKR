import world
import dataloader
import model
import utils
from pprint import pprint
from os.path import join
import os

if world.dataset in ['yelp2018', 'amazon-book', 'ml-20m']:
    dataset = dataloader.Loader(path=join(world.DATA_PATH,world.dataset))
    kg_dataset = dataloader.KGDataset()

print('===========config================')
print(f"PID: {os.getpid()}")
print("cores for test:", world.CORES)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
# print("using bpr loss")
print('===========end===================')

MODELS = {
    'hcmkr': model.HCMKR,
}