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

import utils
import data_loader_v3
import model_v1



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='small_data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'


def train(model, dataloader, optimizer, critierion, metrics, params):
    model.train()

    summ = []
    avg_loss = utils.Running_avg()
    training_step = len(dataloader)

    with tqdm(total=training_step) as t:
        for i, (images_batch, labels_batch) in enumerate(dataloader):

            # using cuda if avaiable
            if params.cuda:
                images_batch = images_batch.cuda(non_blocking=True)
                labels_batch = labels_batch.cuda(non_blocking=True)

            # set images, labels as training variables
            images_batch, labels_batch = Variable(images_batch), Variable(labels_batch)

            output_batch = model(images_batch)
            loss = critierion(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            avg_loss.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
            t.update()

            # compute mean of all metrics in summary
            metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
            logging.info("- Train metrics: " + metrics_string)
            
            del loss


dict = {'apartment': 0, 'church': 1, 'garage': 2, 'house': 3, 'industrial': 4, 'officebuilding': 5, 'retail': 6,
        'roof': 7}

if __name__ == '__main__':
    
    # path = 'small_data'  # change to correct path
    path = 'C:/Users/jack_minimonster/Documents/231n_dataset/BIC_GSV/building_instance_data'
    
    json_path = 'model/params.json'
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = True

    # load data
    dataloaders = data_loader_v3.fetch_dataloader(['train', 'test'], path, params)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    # for i, (image, label) in enumerate(train_dl):
    #     print(label)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = model_v1.resnet50(params).to(device)
    # model = torch.nn.DataParallel(model)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = model_v1.metrics

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train(model, train_dl, optimizer, criterion, metrics, params)


'''
train
mean: (0.4793, 0.4921, 0.4731)
std: (0.0670, 0.0837, 0.1140)

test
mean: (0.4789, 0.4905, 0.4740)
std: (0.2007, 0.2004, 0.2277)
'''