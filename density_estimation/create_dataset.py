import torch
import hyperparams
from torch.utils.data import DataLoader,Dataset
import os
import pickle
from datetime import datetime


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

def load_inputmap(dim_x, args):
    if args.input_type == "discr10":
        dim_u = 20
    elif args.input_type == "polyn3":
        dim_u = 8
    else:
        raise NotImplementedError
    num_inputs = 2 * dim_x + 1 + dim_u
    input_map = {'xe0': torch.arange(0, dim_x),
                 'xref0': torch.arange(dim_x, 2 * dim_x),
                 't': 2 * dim_x,
                 'u_params': torch.arange(2 * dim_x + 1, num_inputs)}
    return input_map, num_inputs

def load_outputmap(dim_x):
    num_outputs = dim_x + 1
    output_map = {'xe': torch.arange(0, dim_x),
                  'rho': dim_x}
    return output_map, num_outputs

def raw2nnData(results_all, args):
    data = []
    input_map = None

    for results in results_all:
        x0 = results['x0']
        t = results['t']
        xref_traj = results['xref_traj']
        xe_traj = results['xe_traj']
        rho_traj = results['rho_traj']
        if 'u_params' not in results.keys():
            uref_traj = results['uref_traj']
            u_params = uref_traj[0, :, ::args.N_u]
            if u_params.shape[1] < 10:
                u_params = torch.cat((u_params[:, :], torch.zeros(u_params.shape[0], 10 - u_params.shape[1])), 1)
        else:
            u_params = results['u_params']

        if input_map is None:
            input_map, num_inputs = load_inputmap(xe_traj.shape[1], args)
            input_tensor = torch.zeros(num_inputs)
            output_map, num_outputs = load_outputmap(xe_traj.shape[1])
            output_tensor = torch.zeros(num_outputs)

        input_tensor[input_map['u_params']] = u_params.flatten()
        input_tensor[input_map['xref0']] = xref_traj[0, :, 0]
        for i_x in range(min(xe_traj.shape[0], 50)):
            input_tensor[input_map['xe0']] = x0[i_x, :] - xref_traj[0, :, 0]
            for i_t in range(0, t.shape[0]):
                input_tensor[input_map['t']] = t[i_t]
                output_tensor[output_map['xe']] = xe_traj[i_x, :, i_t]
                output_tensor[output_map['rho']] = rho_traj[i_x, 0, i_t]
                data.append([input_tensor.numpy().copy(), output_tensor.numpy().copy()])
    return data, input_map, output_map

def nn2rawData(data, input_map, output_map, args):
    input, output = data
    xref0 = input[input_map['xref0']]
    xe0 = input[input_map['xe0']]
    t = input[input_map['t']]
    u_params = input[input_map['u_params']]
    xe = output[output_map['xe']]
    rho = output[output_map['rho']]
    results = {
        'xe0': xe0,
        'xref0': xref0,
        't': t,
        'u_params': u_params,
        'xe': xe,
        'rho': rho,
        }
    return results


if __name__ == "__main__":
    args = hyperparams.parse_args()
    densityData = densityDataset(args)

    print("end")