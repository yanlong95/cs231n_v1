"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model_v5 as net
import data_loader_v5 as data_loader
import visualize


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='dataset/full_region_data',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, critierion, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        critierion: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []

    # compute metrics over the dataset
    for data_batch, labels_type_batch, labels_region_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_type_batch, labels_region_batch = data_batch.cuda(
                non_blocking=True), labels_type_batch.cuda(non_blocking=True), labels_region_batch.cuda(non_blocking=True)
        # fetch the next evaluation batch
        data_batch, labels_type_batch, labels_region_batch = Variable(data_batch), Variable(labels_type_batch), Variable(labels_region_batch)

        # compute model output
        output_type_batch, output_region_batch = model(data_batch)
        loss1 = critierion(output_type_batch, labels_type_batch)
        loss2 = critierion(output_region_batch, labels_region_batch)
        loss = params.alpha * loss1 + (1 - params.alpha) * loss2

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_type_batch = output_type_batch.data.cpu().numpy()
        labels_type_batch = labels_type_batch.data.cpu().numpy()

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_type_batch, labels_type_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        
        # output individual label performance
        class_report = net.classification_report(output_type_batch, labels_type_batch, output_dict=True)
        for key in class_report:
            if len(key) == 1:
                summary_batch['f1score-' + key] = np.float64(class_report[key]['f1-score'])
        
        summ.append(summary_batch)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.resnet50(params, 8).cuda() if params.cuda else net.resnet34(params, 8)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False).to(device)

    critierion = torch.nn.CrossEntropyLoss()
    metrics = net.metrics

    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, critierion, test_dl, metrics, params)
    
    visualize.plot_individual_label_f1score(test_metrics,type='test')
    
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
