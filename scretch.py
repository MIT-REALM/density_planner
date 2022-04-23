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
from train_density import NeuralNetwork
from density_estimation.compute_rawdata import compute_data
from density_estimation.create_dataset import raw2nnData, nn2rawData
import numpy as np


#
# args = hyperparams.parse_args()
# args.device = "cuda" if torch.cuda.is_available() else "cpu"
# model = NeuralNetwork(25, 5, args).to(args.device)
# name_load = '2022-04-21-06-28-01_NN_newCost_CAR_dt10ms_Nsim100_Nu10_iter1000'
# model_params, arg, result = torch.load(('data/trained_nn/' + name_load + '.pt'), map_location=args.device)
# model.load_state_dict(model_params)
#
#
# #results = torch.load('data/trained_nn/2022-04-20-20-29-32_results_CAR_dt10ms_Nsim100_Nu10_iter1000.pt', map_location=args.device)
# #for result in results:
# config = result['hyperparameters']
# name = f"learning_rate={config['learning_rate']}, num_hidden={config['num_hidden']}, size_hidden={config['size_hidden']}, weight_decay={config['weight_decay']}, optimizer={config['optimizer']}"
# plot_losscurves(result['train_loss'], result['test_loss'], name, args)

# args = hyperparams.parse_args()
# args.device = "cuda" if torch.cuda.is_available() else "cpu"
#
# torch.manual_seed(args.random_seed)
# bs = args.batch_size
# system = Car()
# sample_size = 20  # [15, 15, 5, 5] ' number of sampled initial conditions x0
# iteration_number = 10
#
# results_all = compute_data(iteration_number, sample_size, system, args, samples_t=int(np.rint(0.1*args.N_sim)), save=False, plot=False)
# data, input_map, output_map = raw2nnData(results_all, args)
#
# for input_tensor, output_tensor in data:
#
#     model_params, args_NN, result = torch.load(('data/trained_nn/' + "2022-04-21-06-28-01_NN_newCost_CAR_dt10ms_Nsim100_Nu10_iter1000" + '.pt'),
#                                                map_location=args.device)
#     num_inputs = input_tensor.shape[0]
#     model = NeuralNetwork(num_inputs - 1, output_tensor[0], args_NN).to(args.device)
#     model.load_state_dict(model_params)
#     model.eval()









if __name__ == "__main__":
    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.random_seed)
    bs = args.batch_size
    system = Car()
    x = (system.X_MAX - system.X_MIN) * torch.rand(bs, system.DIM_X, 1) + system.X_MIN
    xref = (system.XREF0_MAX - system.XREF0_MIN) * torch.rand(bs, system.DIM_X, 1) + system.XREF0_MIN

    while True:
        uref_traj = system.sample_uref_traj(args.N_sim, args.N_u)  # get random input trajectory
        xref0 = system.sample_xref0()  # sample random xref
        xref_traj = system.compute_xref_traj(xref0, uref_traj, args.N_sim,
                                             args.dt_sim)  # compute corresponding xref trajectory
        xref_traj, uref_traj = system.cut_xref_traj(xref_traj,
                                                    uref_traj)  # cut trajectory where state limits are exceeded
        if xref_traj.shape[2] == args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
            break

    #seen data
    density_data = densityDataset(args)
    train_dataloader = DataLoader(density_data, batch_size=1, shuffle=True)

    args.nameend_dataset = 'datasetVal_files4_CAR_iter100_xSamples20_tSamples10.pickle'
    density_data = densityDataset(args)
    val_dataloader = DataLoader(density_data, batch_size=1, shuffle=True)


    sample_size = 100,

    for params in (params_NN,params_LE, params_MC): #
        x = system.sample_x0(xref0, params["sample_size"])
        rho = torch.ones(x.shape[0], 1, 1)
        if params["title"] == "LE":
            x, rho = system.compute_density(x, xref_traj, uref_traj, rho, N_sim, dt, cutting=False)
            comb = "sum"
        elif params["title"] == "MC":
            with torch.no_grad():
                for i in range(N_sim):
                    x = system.get_next_x(x, xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
            comb = "sum"
        elif params["title"] == "NN":
            model_params, args_NN, result = torch.load(('data/trained_nn/' + params["filename"] + '.pt'), map_location=args.device)
            num_inputs = 26
            model = NeuralNetwork(num_inputs-1, 5, args_NN).to(args.device)
            model.load_state_dict(model_params)
            model.eval()
            u_in = uref_traj[0, :, ::args_NN.N_u]
            if u_in.shape[1] < 10:
                u_in = torch.cat((u_in[:, :], torch.zeros(u_in.shape[0], 10 - u_in.shape[1])), 1)
            input_map = {'rho0': 0,
                         'x0': torch.arange(1, xref_traj.shape[1] + 1),
                         't': xref_traj.shape[1] + 1,
                         'uref': torch.arange(xref_traj.shape[1] + 2, num_inputs)}
            input_tensor = torch.zeros(x.shape[0], num_inputs)
            input_tensor[:, input_map['uref']] = (torch.cat((u_in[0, :], u_in[1, :]), 0)).repeat(x.shape[0],1)
            input_tensor[:, input_map['x0']] = x[:, :, 0]
            input_tensor[:, input_map['rho0']] = rho[:, 0, 0]
            input_tensor[:, input_map['t']] = torch.ones(x.shape[0]) * N_sim * args.dt_sim
            with torch.no_grad():
                input = input_tensor.to(args.device)
                output = model(input[:, 1:])
                x = output[:, 0:4].unsqueeze(-1)
                rho = input[:, 0] * torch.exp(output[:, 4])
                rho = rho.unsqueeze(-1).unsqueeze(-1)
            comb = "mean"

        xe = x - xref_traj

        name = params["title"] + "_time%.2fs_numSim%d_numStates%d_randSeed%d" % (
             args.dt_sim * args.N_sim, args.N_sim, xe.shape[0], args.random_seed)
        #plot_density_heatmap2(xe[:, 0, -1].numpy(), xe[:, 1, -1].numpy(), rho[:, 0, -1].numpy(), name, args, save=False)
        plot_density_heatmap(xe[:, 0, -1].numpy(), xe[:, 1, -1].numpy(), rho[:, 0, -1].numpy(), name, args, combine=comb, save=False)




print("end")

