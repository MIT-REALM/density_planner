import torch
import hyperparams
from torch.utils.data import DataLoader,Dataset
import os
import pickle
import copy
from datetime import datetime
import numpy as np

class densityDataset(Dataset):
    def __init__(self, args):
        self.load_data(args)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor, output_tensor = self.data[index]
        return input_tensor, output_tensor

    def load_data(self, args):
        data = []
        input_map = None
        output_map = None

        if args.load_dataset:
            for file in os.listdir(args.path_dataset):
                if file.endswith(args.nameend_dataset):
                    with open(os.path.join(args.path_dataset, file), "rb") as f:
                        data_all = pickle.load(f)
                    self.data, self.input_map, self.output_map = data_all
                    return
            print("filename args.load_data was not found")
            return

        for file in os.listdir(args.path_rawdata):
            if file.endswith(args.nameend_rawdata):
                with open(os.path.join(args.path_rawdata, file), "rb") as f:
                    results_all = pickle.load(f)
                for results in results_all:
                    t = results['t']
                    uref_traj = results['uref_traj']
                    xref_traj = results['xref_traj']
                    xe_traj = results['xe_traj']
                    rho_traj = results['rho_traj']
                    u_in = uref_traj[0, :, ::args.N_u]

                    if input_map is None:
                        num_inputs = xe_traj.shape[1] + 2 + u_in.shape[0] * u_in.shape[1]
                        input_map = {'x0': torch.arange(0, xe_traj.shape[1]),
                                     'rho0': xe_traj.shape[1],
                                     't': xe_traj.shape[1]+1,
                                     'uref': torch.arange(xe_traj.shape[1] + 2, num_inputs)}
                        input_tensor = torch.zeros(num_inputs)
                        output_map = {'x': torch.arange(0, xe_traj.shape[1]),
                                     'rho': xe_traj.shape[1]}
                        output_tensor = torch.zeros(xe_traj.shape[1] + 1)

                    input_tensor[input_map['uref']] = torch.cat((u_in[0, :], u_in[1, :]), 0)
                    for i_x in range(min(xe_traj.shape[0], 50)):
                        input_tensor[input_map['x0']] = xe_traj[i_x, :, 0] + xref_traj[0, :, 0]
                        input_tensor[input_map['rho0']] = rho_traj[i_x, 0, 0]
                        for i_t in range(1, args.N_sim):
                            input_tensor[input_map['t']] = t[i_t]
                            output_tensor[output_map['x']] = xe_traj[i_x, :, i_t] + xref_traj[0, :, i_t]
                            output_tensor[output_map['rho']] = rho_traj[i_x, 0, i_t]
                            data.append([input_tensor.numpy().copy(), output_tensor.numpy().copy()])

        data_name = args.path_dataset + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + args.nameend_dataset
        with open(data_name, "wb") as f:
            pickle.dump([data, input_map, output_map], f)

        self.data = data
        self.input_map = input_map
        self.output_map = output_map
        return


if __name__ == "__main__":
    args = hyperparams.parse_args()
    densityData = densityDataset(args)

    print("end")