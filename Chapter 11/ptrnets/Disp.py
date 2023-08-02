
import math
from pprint import pprint
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely import geometry

def Disp_results(train_loss,train_accuracy,val_loss,val_accuracy,n_epochs):
    idx_best_train_loss = np.argmin(train_loss.history)
    best_train_loss = train_loss.history[idx_best_train_loss]
    idx_best_train_accuracy = np.argmax(train_accuracy.history)
    best_train_accuracy = train_accuracy.history[idx_best_train_accuracy]
    idx_best_val_loss = np.argmin(val_loss.history)
    best_val_loss = val_loss.history[idx_best_val_loss]
    idx_best_val_accuracy = np.argmax(val_accuracy.history)
    best_val_accuracy = val_accuracy.history[idx_best_val_accuracy]
    print('Best Scores:')
    print(f'train_loss: {best_train_loss:.4f} (ep: {idx_best_train_loss})')
    print(f'train_accuracy {best_train_accuracy:3.2%} (ep: {idx_best_train_accuracy})')
    print(f'val_loss: {best_val_loss:.4f} (ep: {idx_best_val_loss})')
    print(f'val_accuracy: {best_val_accuracy:3.2%} (ep: {idx_best_val_accuracy})')

    x_epochs = list(range(n_epochs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(x_epochs, train_loss.history, 'b')
    ax[0].plot(x_epochs, val_loss.history, 'r')
    _ = ax[0].set_title('Train vs. Val Loss')
    ax[1].plot(x_epochs, train_accuracy.history, 'b', label='Train')
    ax[1].plot(x_epochs, val_accuracy.history, 'r', label='Val')
    _ = ax[1].set_title('Train vs. Val Accuracy')
    ax[1].legend()