"""Main file of the CNN model"""

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
import data_loader_v5 as data_loader
import model_v5 as net
from evaluate_v2 import evaluate
import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default= 'dataset/full_region_data' , help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'


def train(model, dataloader, optimizer, critierion, metrics, params):
    model.train()

    summ = []
    avg_loss = utils.Running_avg()
    training_step = len(dataloader)
    
    with tqdm(total=training_step) as t:
        for i, (images_batch, labels_type_batch, labels_region_batch) in enumerate(dataloader):

            # using cuda if avaiable
            if params.cuda:
                images_batch = images_batch.cuda(non_blocking=True)
                labels_type_batch = labels_type_batch.cuda(non_blocking=True)
                labels_region_batch = labels_region_batch.cuda(non_blocking=True)

            # set images, labels as training variables
            images_batch, labels_type_batch, labels_region_batch = Variable(images_batch), Variable(labels_type_batch), Variable(labels_region_batch)

            output_type_batch, output_region_batch = model(images_batch)
            loss1 = critierion(output_type_batch, labels_type_batch)
            loss2 = critierion(output_region_batch, labels_region_batch)
            loss = params.alpha * loss1 + (1 - params.alpha) * loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_type_batch = output_type_batch.data.cpu().numpy()
                labels_type_batch = labels_type_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric: metrics[metric](output_type_batch, labels_type_batch) for metric in metrics}
                summary_batch['loss'] = np.float64(loss.item())
                
                # output individual label performance
                class_report = net.classification_report(output_type_batch, labels_type_batch, output_dict=True)
                for key in class_report:
                    if len(key) == 1:
                        # type label: {'apartment': 0, 'church': 1, 'house': 2, 'industrial': 3, 'officebuilding': 4, 'retail': 5, 'roof': 6}
                        summary_batch['f1score-' + key] = np.float64(class_report[key]['f1-score'])
                
                summ.append(summary_batch)
                
            avg_loss.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(avg_loss()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    return metrics_mean

    
    
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, critierion, metrics, params, model_dir,
                       restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)
        
    best_val_acc = 0.0
    best_val_metrics = []
    learning_rate_0 = params.learning_rate
    train_acc_series = []
    val_acc_series = []
    train_loss_series = []
    
    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        # train model
        train_metrics = train(model, train_dataloader, optimizer, critierion, metrics, params)
        
        # learning rate exponential decay
        params.learning_rate = learning_rate_0 * np.exp(-params.exp_decay_k * epoch)
        
        # evaluate
        val_metrics = evaluate(model, critierion, val_dataloader, metrics, params)
        
        # find accuracy from validation dataset
        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_val_acc
        
        # save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_dir)
        
        # save accuracy / loss to array for plot
        train_acc_series.append(train_metrics['accuracy'])
        val_acc_series.append(val_metrics['accuracy'])
        train_loss_series.append(train_metrics['loss'])
        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_val_metrics = val_metrics

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        
        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        print('******************************************')
    
    # plot visualized performance
    visualize.plot_train_val_accuracy(train_acc_series, val_acc_series)
    visualize.plot_loss(train_loss_series)
    # save best validation F1 score plot
    visualize.plot_individual_label_f1score(best_val_metrics)

# load the class label
# file_name = 'categories_places365.txt'
# if not os.access(file_name, os.W_OK):
#     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
#     os.system('wget ' + synset_url)
# classes = list()
# with open(file_name) as class_file:
#     for line in class_file:
#         classes.append(line.strip().split(' ')[0][3:])
# classes = tuple(classes)

dict_type = {'apartment': 0, 'church': 1, 'house': 2, 'industrial': 3, 'officebuilding': 4, 'retail': 5,
        'roof': 6}
dict_region = {'ca': 0, 'ne': 1, 'se': 2}

if __name__ == '__main__':
    args = parser.parse_args()
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    
    # true if use GPU
    params.cuda = torch.cuda.is_available()

    # turn alpha (a*L1 + (1-a)*L2)
    params.alpha = 0.7
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info("Loading the datasets...")

    # load data
    dataloaders = data_loader.fetch_dataloader(['train', 'test'], args.data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']
    
    logging.info("- done.")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # change CNN architecture
    model = net.resnet50(params, 8).to(device)
    # model = net.vgg16(params, 8).to(device)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=False).to(device)
    # model = torch.nn.DataParallel(model)
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = net.metrics
    

    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, criterion, metrics, params, args.model_dir,
                    args.restore_file)


'''
train
mean: (0.4793, 0.4921, 0.4731)
std: (0.0670, 0.0837, 0.1140)

test
mean: (0.4789, 0.4905, 0.4740)
std: (0.2007, 0.2004, 0.2277)
'''