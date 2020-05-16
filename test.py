import argparse
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import data_loader
import utils

path = 'small_data'             # change to correct path
json_path = 'params.json'
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
params.cuda = 'cpu'

# load data
dataloaders = data_loader.fetch_dataloader(['train'], path, params)
train_dl = dataloaders['train']
print(len(train_dl))



