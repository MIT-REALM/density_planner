from systems.sytem_CAR import Car
import torch
import hyperparams
from plots.plot_functions import plot_scatter, plot_density_heatmap, plot_ref
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
import numpy as np
import os
from motion_planning.utils import make_path
from data_generation.utils import load_outputmap, get_input_tensors, get_output_variables
from datetime import datetime



if __name__ == "__main__":
    sample_size = 5000
    args = hyperparams.parse_args()

    # data preparation
    torch.manual_seed(args.random_seed)
    system = Car()
    name = '3layer_dist%.1f-%.1f_%s' % (system.d_max, system.d_max2, args.input_type)
    path = make_path(args.path_plot_references, name)

    for k in range(40):
        xref_traj, rho_traj, uref_traj, u_params, xe_traj, t_vec = system.get_valid_trajectories(sample_size, args)
        plot_ref(xref_traj, uref_traj, "traj%d" % k, args, system, t=t_vec, x_traj=xe_traj[:100, :, :]+xref_traj,
                 include_date=True, folder=path)
        plot_density_heatmap("traj%d_iter50" % k, args, xe_le=xe_traj[:, :, [50]], rho_le=rho_traj[:, :, [50]],
                             include_date=True, folder=path)
        plot_density_heatmap("traj%d_iter80" % k, args, xe_le=xe_traj[:, :, [70]], rho_le=rho_traj[:, :, [70]],
                             include_date=True, folder=path)
        plot_density_heatmap("traj%d_iter%d" % (k, xe_traj.shape[2]), args, xe_le=xe_traj[:, :, [-1]], rho_le=rho_traj[:, :, [-1]],
                             include_date=True, folder=path)