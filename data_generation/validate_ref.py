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
from data_generation.pde_solver import Solver_PDE


if __name__ == "__main__":
    sample_size_le = 5000
    sample_size_mc = 0
    fpe_fourier = True
    args = hyperparams.parse_args()

    # data preparation
    torch.manual_seed(args.random_seed)
    system = Car(args)
    solver = Solver_PDE(system, args)
    name = 'dist%.1f-%.1f_%s_samples%d-%d' % (system.DIST[0, 0, 0], system.DIST[0, 1, 1], args.input_type,
                                              sample_size_le, sample_size_mc)
    path = make_path(args.path_plot_references, name)
    log_density = True
    time_steps = [30, 50, 70, 100, 200, 300, 500, 800]
    for k in range(40):

        while True:
            uref_traj, u_params = system.sample_uref_traj(args)  # get random input trajectory
            xref0 = system.sample_xref0()  # sample random xref
            xref_traj = system.compute_xref_traj(xref0, uref_traj, args)  # compute corresponding xref trajectory
            xref_traj, uref_traj = system.cut_xref_traj(xref_traj, uref_traj)  # cut trajectory where state limits are exceeded
            if xref_traj.shape[2] > 0.8 * args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
                break

        # numerical solution of fpe
        if fpe_fourier:
            with torch.no_grad():
                rho_fpe = solver.solve_fpe(uref_traj, xref_traj, time_steps, args, folder=path)

        # LE:
        if sample_size_le is not None:
            x0 = system.sample_x0(xref0, sample_size_le)  # get random initial states
            if log_density:
                rho0 = torch.zeros(x0.shape[0], 1, 1)  # equal initial density
            else:
                rho0 = torch.ones(x0.shape[0], 1, 1)  # equal initial density
            x_le, rho_le = system.compute_density(x0, xref_traj, uref_traj, rho0, xref_traj.shape[2],
                                                    args.dt_sim, cutting=False, log_density=log_density)
            xe_le = x_le - xref_traj
            plot_ref(xref_traj, uref_traj, "traj%d" % k, args, system, x_traj=xe_le[:100, :, :]+xref_traj,
                 include_date=True, folder=path)

        # MC
        if sample_size_mc is not None:
            if sample_size_mc != 0:
                x0 = system.sample_x0(xref0, sample_size_mc)
                x_mc, _ = system.compute_density(x0, xref_traj, uref_traj, None, xref_traj.shape[2],
                                                        args.dt_sim, cutting=False, computeDensity=False)
                if sample_size_le is not None:
                    x_mc = torch.cat((x_le, x_mc), 0)
            else:
                x_mc = x_le
            xe_mc = x_mc - xref_traj

        for i, iter_plot in enumerate(time_steps): #50, 70, xe_traj.shape[2]-1]:
            xe_dict = {}
            rho_dict = {}
            if sample_size_le is not None:
                xe_dict["LE"] = xe_le[:, :, [iter_plot]]
                rho_dict["LE"] = rho_le[:, :, [iter_plot]]
            if sample_size_mc is not None:
                xe_dict["MC"] = xe_mc[:, :, [iter_plot]]
                rho_dict["MC"] = torch.ones(xe_mc.shape[0], 1, 1) / xe_mc.shape[0]
            if fpe_fourier:
                xe_dict["FPE"] = solver.xe
                rho_dict["FPE"] = rho_fpe[i]
            plot_density_heatmap("traj%d_iter%d" % (k, iter_plot), args, xe_dict, rho_dict, system=system,
                             include_date=True, folder=path, log_density=log_density)


