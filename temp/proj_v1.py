import argparse
import logging
import os
import time
from tqdm import tqdm

import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from torch.autograd import Variable

import data_loader
import utils
import model_v1
from evaluate import evaluate



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'



"""
    define a train process
    args: train dataloader
"""
def train(model, dataloader, optimizer, criterion, metrics, params):
    model.train()

    summ = []
    avg_loss = utils.Running_avg()
    training_step = len(dataloader)

    with tqdm(total=training_step) as t:
        for i, (train_batch_orig, labels_batch) in enumerate(dataloader):

            # using cuda if available
            # if params_training['cuda']:
            if params.cuda:
                train_batch_orig = train_batch_orig.cuda(non_blocking=True)
                labels_batch = labels_batch.cuda(non_blocking=True)

            # load variables
            train_batch_orig, labels_batch = Variable(train_batch_orig), Variable(labels_batch)
            train_batch = train_batch_orig / 255.0

            output_batch = model(train_batch)
            loss = criterion(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            # if i % params_training['save_summary_steps'] == 0:
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


def train_and_evaluate(model, dataloaders, optimizer, criterion, metrics, model_dir, params, restore_file=None):
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    best_acc = 0.0
    # for epoch in range(params_training['num_epoch']):
    for epoch in range(params.num_epochs):
        # logging.info("Epoch {}/{}".format(epoch + 1, params_training['num_epoch']))
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        train(model, train_dl, optimizer, criterion, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, val_dl, criterion, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc >= best_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        if is_best:
            logging.info("- Found new best accuracy")
            best_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # path = 'data'
    # path = 'data_test'
    # path = 'data_test_bugholes'
    path = 'data_pretrain'
    train_data = os.path.join(path, 'train_mix')
    evl_data = os.path.join(path, 'val_mix')
    test_data = os.path.join(path, 'test_mix')

    start = time.time()

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    params.cuda = use_cuda

    model = model_v1.resnet50(params).to(device)
    # model = model_v1.resnet18(params).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    metrics = model_v1.metrics

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], path, params)
    logging.info("- done.")

    # logging.info("Starting training for {} epochs".format(params_training['num_epoch']))
    logging.info("Starting training for {} epochs".format(params.num_epochs))

    train_and_evaluate(model, dataloaders, optimizer, criterion, metrics, args.model_dir, params)

    end = time.time()
    print('Training Time: ', end-start)