# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:26:42 2020

@author: jack_minimonster
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_train_val_accuracy(train_acc_series, val_acc_series):
    
    train_acc_series = np.array(train_acc_series)
    val_acc_series = np.array(val_acc_series)
    epoch_series = range(1, len(train_acc_series)+1, 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_series, train_acc_series, 'k-', epoch_series, val_acc_series, 'k--')
    ax.legend(('train accuracy', 'validation accuracy'), loc='best')
    ax.set(title="Accuracy",
           ylabel='accuracy',
           xlabel='epochs',
           ylim=[0,1.0])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()
    fig.savefig('train_val_accuracy.png', dpi=fig.dpi)
    
    
def plot_loss(loss_series):
    loss_series = np.array(loss_series)
    epoch_series = range(1, len(loss_series)+1, 1)
    
    fig, ax = plt.subplots()
    ax.plot(epoch_series, loss_series, 'k-')
    ax.set(title="Average Loss",
           ylabel='loss',
           xlabel='epochs')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.show()
    fig.savefig('train_loss.png', dpi=fig.dpi)
    
    
    
def plot_individual_label_f1score(val_metrics, type='val'):

    classes = ['apartments', 'church', 'house', 'industrial', 'office', 'retail', 'roof']
    f1score = []
    for key in val_metrics:
        if 'f1score' in key:
            f1score.append(val_metrics[key])
    
    fig, ax = plt.subplots()
    ax.bar(classes, f1score)
    ax.set(title="Best Validation F1 Score",
           ylabel='F1 Score',
           xlabel='Building Categories',
           ylim=[0, 1])
    plt.xticks(rotation=90)
    # plt.show()
    fig.savefig('f1score_' + type + '.png', dpi=fig.dpi)