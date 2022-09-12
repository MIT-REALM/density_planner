from systems.sytem_CAR import Car
import torch
import hyperparams
from plots.plot_functions import plot_density_heatmap, plot_ref
from motion_planning.utils import make_path
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap


"""
script to compare density predictions of LE, Monte Carlo simulation and neural density predictor
"""
if __name__ == "__main__":

    use_le = True # compute density with Liouville equation
    use_mc = False  # compute density with Monte Carlo
    use_nn = True  # compute density with NN
    use_nn2 = False  # compute density with second NN

    sample_size = 1000  # number of samples
    sample_size_mc = 0  # number of additional samples for Monte Carlo simulation
    args = hyperparams.parse_args()

    # preparation
    torch.manual_seed(args.random_seed)
    system = Car(args)
    system.X_MIN[0, 0, 0] = -20
    system.X_MIN[0, 1, 0] = -20
    system.X_MAX[0, 0, 0] = 20
    system.X_MAX[0, 1, 0] = 20
    name = "figures_paper"
    path = make_path(args.path_plot_densityheat, name)

    if use_nn:
        _, num_inputs = load_inputmap(system.DIM_X, args)
        _, num_outputs = load_outputmap(system.DIM_X, args)
        model, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True)
        model.eval()
    if use_nn2:
        _, num_inputs = load_inputmap(system.DIM_X, args)
        _, num_outputs = load_outputmap(system.DIM_X, args)
        model2, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True, nn2=True)
        model2.eval()

    log_density = True
    time_steps = [30, 50, 70, 100, 200, 300, 400, 500, 800, 900, 1000]
    for k in range(12):
        xe0 = system.sample_xe0(sample_size)
        Up, uref_traj, xref_traj = system.get_valid_ref(args)

        # LE:
        if use_le:  # get random initial states
            xe_le, rho_le = system.compute_density(xe0, xref_traj, uref_traj, args.dt_sim,
                                                  cutting=False, log_density=log_density)
            plot_ref(xref_traj, uref_traj, "traj%d" % k, args, system, x_traj=xe_le[:50, :, :]+xref_traj,
                 include_date=True, folder=path)

        #MC
            if sample_size_mc != 0:
                xe0 = system.sample_xe0(sample_size_mc)
                xe_mc, _ = system.compute_density(xe0, xref_traj, uref_traj, args.dt_sim,
                                                 cutting=False, compute_density=False)
                if use_le:
                    xe_mc = torch.cat((xe_le, xe_mc), 0)
            else:
                xe_mc = xe_le

        if use_nn:
            t_vec = args.dt_sim * torch.arange(0, xref_traj.shape[2])
            xe_nn = torch.zeros(xe0.shape[0], system.DIM_X, xref_traj.shape[2])
            rho_nn = torch.zeros(xe0.shape[0], 1, xref_traj.shape[2])
            for i, t in enumerate(t_vec):
                xe_nn[:, :, [i]], rho_nn[:, :, [i]] = get_nn_prediction(model, xe0, xref_traj[0, :, 0], t, Up, args)

        if use_nn2:
            t_vec = args.dt_sim * torch.arange(0, xref_traj.shape[2])
            xe_nn2 = torch.zeros(xe0.shape[0], system.DIM_X, xref_traj.shape[2])
            rho_nn2 = torch.zeros(xe0.shape[0], 1, xref_traj.shape[2])
            for i, t in enumerate(t_vec):
                xe_nn2[:, :, [i]], rho_nn2[:, :, [i]] = get_nn_prediction(model2, xe0, xref_traj[0, :, 0], t, Up, args)

        for i, iter_plot in enumerate(time_steps): #50, 70, xe_traj.shape[2]-1]:
            xe_dict = {}
            rho_dict = {}
            if use_le:
                xe_dict["LE"] = xe_le[:, :, [iter_plot]]
                rho_dict["LE"] = rho_le[:, :, [iter_plot]]
            if use_mc:
                xe_dict["MC"] = xe_mc[:, :, [iter_plot]]
                rho_dict["MC"] = torch.ones(xe_mc.shape[0], 1, 1) / xe_mc.shape[0]
            if use_nn:
                xe_dict["NN"] = xe_nn[:, :, [iter_plot]]
                rho_dict["NN"] = rho_nn[:, :, [iter_plot]]
            if use_nn2:
                xe_dict["NN2"] = xe_nn2[:, :, [iter_plot]]
                rho_dict["NN2"] = rho_nn2[:, :, [iter_plot]]

            plot_density_heatmap("traj%d_iter%d" % (k, iter_plot), args, xe_dict, rho_dict, system=system,
                             include_date=True, folder=path, log_density=log_density)
