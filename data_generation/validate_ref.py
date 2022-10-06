from systems.sytem_CAR import Car
import torch
import hyperparams
from plots.plot_functions import plot_ref
from motion_planning.utils import make_path


"""
script to plot the reference trajectory and the state trajectories which result from applying the tracking controller
"""
if __name__ == "__main__":
    sample_size = 1000  # number of samples
    sample_size_mc = 0  # number of additional samples for Monte Carlo simulation
    args = hyperparams.parse_args()

    # preparation
    torch.manual_seed(args.random_seed)
    system = Car(args)
    name = "figures_paper"
    path = make_path(args.path_plot_references, name)

    log_density = True
    time_steps = [30, 50, 70, 100, 200, 300, 400, 500, 800, 900, 1000]
    for k in range(12):
        xe0 = system.sample_xe0(sample_size)
        Up, uref_traj, xref_traj = system.get_valid_ref(args)

        xe_le, rho_le = system.compute_density(xe0, xref_traj, uref_traj, args.dt_sim,
                                                  cutting=False, log_density=log_density)
        plot_ref(xref_traj, uref_traj, "traj%d" % k, args, system, x_traj=xe_le[:50, :, :]+xref_traj,
                 include_date=True, folder=path)




