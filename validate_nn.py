import torch
from systems.sytem_CAR import Car
from systems.ControlAffineSystem import ControlAffineSystem
import sys
from utils import approximate_derivative
import matplotlib.pyplot as plt
import torch
import hyperparams
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from density_estimation.create_dataset import densityDataset
from datetime import datetime
from plot_functions import plot_losscurves
import os
from train_density import test, load_dataloader, load_nn, loss_function
from density_estimation.compute_rawdata import compute_data
from density_estimation.create_dataset import raw2nnData, nn2rawData
import numpy as np


if __name__ == "__main__":

    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = 'finalTrain'
    results = []

    short_name = run_name
    nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'
    print(name)

    train_dataloader, validation_dataloader = load_dataloader(args)
    model, optimizer = load_nn(len(train_dataloader.dataset.data[0][0]), len(train_dataloader.dataset.data[0][1]),
                               args)
