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


def raw2nnData(results_all, args):
    data = []
    input_map = None

    for results in results_all:
        rho0 = results['rho0']
        x0 = results['x0']
        t = results['t']
        uref_traj = results['uref_traj']
        xref_traj = results['xref_traj']
        xe_traj = results['xe_traj']
        rho_traj = results['rho_traj']
        u_in = uref_traj[0, :,::args.N_u]# just save every N_u'th input (assume input stays constant for N_u timesteps)
        if u_in.shape[1] < 10:
            u_in = torch.cat((u_in[:, :], torch.zeros(u_in.shape[0], 10 - u_in.shape[1])), 1)

        if input_map is None:
            num_inputs = 2 * xe_traj.shape[1] + 2 + u_in.shape[0] * u_in.shape[1]
            input_map = {'rho0': 0,
                         'x0': torch.arange(1, xe_traj.shape[1] + 1),
                         'xref0': torch.arange(xe_traj.shape[1] + 1, 2 * xe_traj.shape[1] + 1),
                         't': 2 * xe_traj.shape[1] + 1,
                         'uref': torch.arange(2 * xe_traj.shape[1] + 2, num_inputs)}
            input_tensor = torch.zeros(num_inputs)
            output_map = {'x': torch.arange(0, xe_traj.shape[1]),
                          'rho': xe_traj.shape[1]}
            output_tensor = torch.zeros(xe_traj.shape[1] + 1)

        input_tensor[input_map['uref']] = torch.cat((u_in[0, :], u_in[1, :]), 0)
        input_tensor[input_map['xref0']] = xref_traj[0, :, 0]
        for i_x in range(min(xe_traj.shape[0], 50)):
            input_tensor[input_map['x0']] = x0[i_x, :]
            input_tensor[input_map['rho0']] = rho0[i_x, 0]
            for i_t in range(0, t.shape[0]):
                input_tensor[input_map['t']] = t[i_t]
                output_tensor[output_map['x']] = xe_traj[i_x, :, i_t] + xref_traj[0, :, i_t]
                output_tensor[output_map['rho']] = rho_traj[i_x, 0, i_t]
                data.append([input_tensor.numpy().copy(), output_tensor.numpy().copy()])
    return data, input_map, output_map

def nn2rawData(data, input_map, output_map, args):
    input, output = data
    xref0 = input[input_map['xref0']]
    x0 = input[input_map['x0']]
    rho0 = input[input_map['rho0']]
    t = input[input_map['t']]
    u_in = input[input_map['uref']]
    # uref = torch.ones(1, self.DIM_U, N_sim)
    # for i in range(math.ceil(N_sim / N_u)):
    #     interv_end = min((i + 1) * N_u, N_sim)
    #     u[0, :, i * N_u: interv_end] = (self.UREF_MAX - self.UREF_MIN) * torch.rand(1, self.DIM_U, 1) + self.UREF_MIN
    x = output[output_map['x']]
    rho = output[output_map['rho']]
    results = {
        'x0': x0,
        'xref0': xref0,
        'rho0': rho0,
        't': t,
        'u_in': u_in,
        'x': x,
        'rho': rho,
        }
    return results


if __name__ == "__main__":
    args = hyperparams.parse_args()
    densityData = densityDataset(args)

    print("end")