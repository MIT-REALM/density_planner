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
from plot_functions import plot_losscurves, plot_scatter
import os
from train_density import load_dataloader, load_nn, loss_function
from density_estimation.compute_rawdata import compute_data
from density_estimation.create_dataset import raw2nnData, nn2rawData
import numpy as np


def get_nn_prediction(model, rho0, xe0, xref0, t, uref_traj, num_inputs, args):
    u_in = uref_traj[0, :, ::args.N_u]
    if u_in.shape[1] < 10:
        u_in = torch.cat((u_in[:, :], torch.zeros(u_in.shape[0], 10 - u_in.shape[1])), 1)
    input_map = {'rho0': 0,
                 'xe0': torch.arange(1, xref0.shape[0] + 1),
                 'xref0': torch.arange(xref0.shape[0] + 1, 2 * xref0.shape[0] + 1),
                 't': 2 * xref0.shape[0] + 1,
                 'uref': torch.arange(2 * xref0.shape[0] + 2, num_inputs)}

    input_tensor = torch.zeros(xe0.shape[0], num_inputs)
    input_tensor[:, input_map['uref']] = (torch.cat((u_in[0, :], u_in[1, :]), 0)).repeat(xe0.shape[0], 1)
    input_tensor[:, input_map['xe0']] = xe0
    input_tensor[:, input_map['rho0']] = rho0
    input_tensor[:, input_map['t']] = t * torch.ones(xe0.shape[0])
    input_tensor[:, input_map['xref0']] = xref0
    with torch.no_grad():
        input = input_tensor.to(args.device)
        output = model(input[:, 1:])
        xe = output[:, 0:4].unsqueeze(-1)
        rho = input[:, 0] * torch.exp(output[:, 4])
        rho = rho.unsqueeze(-1).unsqueeze(-1)
    return xe, rho

if __name__ == "__main__":
    sample_size = 100
    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = 'finalTrain'
    results = []

    #short_name = run_name
    #nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'

    # data preparation
    torch.manual_seed(args.random_seed)
    bs = args.batch_size
    system = Car()
    valid = False
    while not valid:
        uref_traj = system.sample_uref_traj(args.N_sim, args.N_u)  # get random input trajectory
        xref0 = system.sample_xref0()  # sample random xref
        xref_traj = system.compute_xref_traj(xref0, uref_traj, args.N_sim, args.dt_sim)  # compute corresponding xref trajectory
        xref_traj, uref_traj = system.cut_xref_traj(xref_traj, uref_traj)  # cut trajectory where state limits are exceeded
        if xref_traj.shape[2] != args.N_sim:  # start again if reference trajectory is shorter than N_sim
            continue
        x0 = system.sample_x0(xref0, sample_size)
        rho0 = 1 / (torch.prod(system.X_MAX-system.X_MIN)) * torch.ones(x0.shape[0], 1, 1)

        # LE prediction
        x_traj, rho_traj = system.compute_density(x0, xref_traj, uref_traj, rho0, xref_traj.shape[2],
                                                  args.dt_sim)  # compute x and rho trajectories
        if rho_traj.dim() < 2 or x_traj.shape[2] < args.N_sim:  # start again if x trajectories shorter than N_sim
            continue
        valid = True
    xe_LE = x_traj - xref_traj

    # NN prediction
    num_inputs = 30
    model, _ = load_nn(num_inputs, 5, args)
    model.eval()
    step = 5
    t_vec = np.arange(0, args.N_sim * args.dt_sim, step * args.dt_sim)
    for i, t in enumerate(t_vec):
        xe_nn, rho_nn = get_nn_prediction(model, rho_traj[:, 0, 0], x_traj[:, :, 0] - xref_traj[0, :, 0], xref_traj[0, :, 0],
                                          t, uref_traj, num_inputs, args)
        error = xe_nn[:, :, 0] - xe_LE[:, :, i*step]
        print("Max error: %.2f, Mean error: %.2f, MSE: %.2f" %
              (torch.max(torch.abs(error)), torch.mean(torch.abs(error)), torch.mean(torch.abs(error ** 2))))
        plot_scatter(xe_nn[:,:,0], xe_LE[:, :, i*step], rho_nn[:,0,0], rho_traj[:, 0, i*step],
                     run_name + "_time=%.2fs" % t, args, save=True, show=True, filename=None)





    # u_in = uref_traj[0, :, ::args_NN.N_u]
    # if u_in.shape[1] < 10:
    #     u_in = torch.cat((u_in[:, :], torch.zeros(u_in.shape[0], 10 - u_in.shape[1])), 1)
    # input_map = {'rho0': 0,
    #              'xe0': torch.arange(1, xref_traj.shape[1] + 1),
    #              'xref0': torch.arange(xref_traj.shape[1] + 1, 2 * xref_traj.shape[1] + 1),
    #              't': 2 * xref_traj.shape[1] + 1,
    #              'uref': torch.arange(2 * xref_traj.shape[1] + 2, num_inputs)}
    # input_tensor = torch.zeros(x.shape[0], num_inputs)
    # input_tensor[:, input_map['uref']] = (torch.cat((u_in[0, :], u_in[1, :]), 0)).repeat(x.shape[0], 1)
    # input_tensor[:, input_map['xe0']] = x[:, :, 0] - xref_traj[0, :, 0]
    # input_tensor[:, input_map['rho0']] = rho[:, 0, 0]
    # input_tensor[:, input_map['t']] = torch.ones(x.shape[0]) * N_sim * args.dt_sim
    # input_tensor[:, input_map['xref0']] = xref_traj[0, :, 0]
    # with torch.no_grad():
    #     input = input_tensor.to(args.device)
    #     output = model(input[:, 1:])
    #     xe = output[:, 0:4].unsqueeze(-1)
    #     rho = input[:, 0] * torch.exp(output[:, 4])
    #     rho = rho.unsqueeze(-1).unsqueeze(-1)


