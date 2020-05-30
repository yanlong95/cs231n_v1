# -*- coding: utf-8 -*-
"""
Single Image Classification Confidence Tester
Created on Sat May 30 11:58:01 2020

@author: jack_minimonster
"""


import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms
from torch.nn import functional as F
import model_v4 as net
import utils
import os
import argparse
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")
                     
# Load the parameters
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(
    json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

# use GPU if available
params.cuda = True     # use GPU is available

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# define model architecture
model = net.resnet50(params, 8).to(device)
 
checkpoint = torch.load(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# evl and test transformer
eval_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.4789, 0.4905, 0.4740), (0.2007, 0.2004, 0.2277))])

# load the class label
classes = ('apartment', 'church', 'garage', 'house', 'industrial', 'office', 'retail', 'roof')

# load the test image
img_name = 'test.jpg'

img = Image.open(img_name)
input_img = V(eval_transformer(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img.cuda())
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('net prediction on {}'.format(img_name))
# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))