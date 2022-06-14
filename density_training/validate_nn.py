from systems.sytem_CAR import Car
import torch
import hyperparams
from plots.plot_functions import plot_scatter, plot_density_heatmap, plot_ref
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
import numpy as np
import os
from data_generation.utils import load_outputmap, get_input_tensors, get_output_variables
from datetime import datetime



if __name__ == "__main__":
    sample_size = 50
    args = hyperparams.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    run_name = args.run_name
    results = []

    #short_name = run_name
    #nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'

    # data preparation
    torch.manual_seed(args.random_seed)
    bs = args.batch_size
    system = Car()
    for k in range(40):
        xref_traj, rho_traj, uref_traj, u_params, xe_traj, t_vec = system.get_valid_trajectories(sample_size, args)
        plot_ref(xref_traj, uref_traj, 'dist0.7', args, system, t=t_vec, x_traj=xe_traj[:50, :, :]+xref_traj, include_date=True)
    #x_traj = xe_traj + xref_traj

    # NN prediction
    _, num_inputs = load_inputmap(xref_traj.shape[1], args)
    _, num_outputs = load_outputmap(xref_traj.shape[1])
    model, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True)
    model.eval()
    #step = 5
    #t_vec = np.arange(0, args.N_sim * args.dt_sim, step * args.dt_sim)
    xe_nn = torch.zeros_like(xe_traj)
    rho_nn = torch.zeros_like(rho_traj)

    args.path_plot_densityheat = os.path.join(args.path_plot_densityheat, datetime.now().strftime("%Y-%m-%d") + "_" + args.run_name + "/")
    if not os.path.exists(args.path_plot_densityheat):
        os.makedirs(args.path_plot_densityheat)
    args.path_plot_scatter = os.path.join(args.path_plot_scatter, datetime.now().strftime("%Y-%m-%d") + "_" + args.run_name + "/")
    if not os.path.exists(args.path_plot_scatter):
        os.makedirs(args.path_plot_scatter)

    for i, t in enumerate(t_vec):
        xe_nn[:,:, [i]], rho_nn[:, :, [i]] = get_nn_prediction(model, xe_traj[:, :, 0], xref_traj[0, :, 0],
                                          t, u_params, args)
        error = torch.sqrt((xe_nn[:, 0, i] - xe_traj[:, 0, i]) ** 2 + (xe_nn[:, 1, i] - xe_traj[:, 1, i]) ** 2)
        print("Max position error: %.3f, Mean position error: %.4f" %
              (torch.max(torch.abs(error)), torch.mean(torch.abs(error))))
        min_rho = torch.minimum(rho_nn[:, 0, i].min(), rho_traj[:, 0, i].min())
        max_rho = torch.maximum(rho_nn[:, 0, i].max(), rho_traj[:, 0, i].max())
        plot_limits = [min_rho, max_rho]
        # plot_density_heatmap2(xe_nn[:, :, i], rho_nn[:, 0, i], "time=%.2fs_NN" % t, args, plot_limits=plot_limits,
        #                      save=True, show=True, filename=None)
        # plot_density_heatmap2(xe_traj[:, :, i], rho_traj[:, 0, i], "time=%.2fs_LE" % t, args, plot_limits=plot_limits,
        #                      save=True, show=False, filename=None)
        with torch.no_grad():
            plot_density_heatmap("Time=%.2fs" % t, args, xe_le=xe_traj[:, :, [i]], rho_le=rho_traj[:, :, [i]],
                              xe_nn=xe_nn[:, :, [i]], rho_nn=rho_nn[:, :, [i]],
                              save=True, show=False, filename=None)
            plot_scatter(xe_nn[:50,:,i], xe_traj[:50, :, i], rho_nn[:50,0,i], rho_traj[:50, 0, i],
                            "Time=%.2fs" % t, args, save=True, show=False, filename=None, weighted=False)







