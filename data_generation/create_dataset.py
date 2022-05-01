import torch
import hyperparams
from torch.utils.data import DataLoader,Dataset
import os
import pickle
from datetime import datetime
from systems.sytem_CAR import Car
from utils import *

class densityDataset(Dataset):
    """
    dataset class for training the neural density estimator
    """

    def __init__(self, args, mode=None):
        self.load_data(args, mode)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor, output_tensor = self.data[index]
        return input_tensor, output_tensor

    def load_data(self, args, mode):

        if mode is not None:
            if mode == "Val":
                filename_data = args.nameend_ValDataset
            elif mode == "Train":
                filename_data = args.nameend_TrainDataset
            # load dataset from specified path args.path_dataset
            for file in os.listdir(args.path_dataset):
                if file.endswith(filename_data): # just consider the first file with specified filename
                    with open(os.path.join(args.path_dataset, file), "rb") as f:
                        data_all = pickle.load(f)
                    self.data, self.input_map, self.output_map = data_all
                    return
            print("filename_data was not found")
            return

        # create new dataset from raw density data

        data_allFiles = []
        i = 0
        for file in os.listdir(args.path_rawdata):
            if file.endswith(args.nameend_rawdata): # just consider data with specified filename
                with open(os.path.join(args.path_rawdata, file), "rb") as f:
                    results_all, _ = pickle.load(f)

                # reformat the data
                if len(results_all) != 0:
                    data, input_map, output_map = raw2nnData(results_all, args)
                    data_allFiles += data
                i += 1

        # save data
        data_name = args.path_dataset + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_dataset_files%d' % i + args.nameend_rawdata
        with open(data_name, "wb") as f:
            pickle.dump([data_allFiles, input_map, output_map], f)

        self.data = data_allFiles
        self.input_map = input_map
        self.output_map = output_map
        return

if __name__ == "__main__":
    args = hyperparams.parse_args()
    densityData = densityDataset(args)

    print("end")