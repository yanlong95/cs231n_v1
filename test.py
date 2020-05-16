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

import data_loader_v1
import utils

import data_loader_v3
import data_loader_v2

path = 'small_data'             # change to correct path
json_path = 'params.json'
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)
params.cuda = 'cpu'

dict = {'apartment': 0, 'church': 1, 'garage': 2, 'house': 3, 'industrial': 4, 'officebuilding': 5, 'retail': 6,
        'roof': 7}

if __name__ == '__main__':
    path = 'BIC_GSV'  # change to correct path
    # load data
    dataloaders = data_loader_v3.fetch_dataloader(['Building_labeled_train_data', 'Building_labeled_test_data'], path, params)
    train_dl = dataloaders['Building_labeled_train_data']
    test_dl = dataloaders['Building_labeled_test_data']
    length = 0
    for i, (image, label) in enumerate(train_dl):
        length += len(image)
        print(length)
