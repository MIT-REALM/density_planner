from motion_planning.utils import sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle, MPOptimizer
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from systems.sytem_CAR import Car
import pickle
import time
from systems.utils import get_mesh_pos
from plots.plot_functions import plot_grid

if __name__ == '__main__':
    args = hyperparams.parse_args()
    torch.manual_seed(args.random_seed)

    # 0. create environment
    env = create_environment(["street", "obstacleBottom", "pedLR"], args)
    #plot_grid(env, args)
    env.forward_occupancy(step_size=100)
    env.enlarge_shape()
    #plot_grid(env, args)
    #plot_grid(env.grid_enlarged, args, timestep=0)
    #plot_grid(env.grid_enlarged, args)

    system = Car()
    pdf0 = sample_pdf(system, 5)
    xref0 = torch.tensor([0, -25, 1.5, 3]).reshape(1, -1, 1).type(torch.FloatTensor)
    xrefN = torch.tensor([0, 0, 4, 1]).reshape(1, -1, 1) #, [10, 5, 10, 6]]).reshape(2, -1, 1)
    t_vec = torch.arange(0, args.dt_sim * args.N_sim - 0.001, args.dt_sim * args.factor_pred)
    _, u_params = system.sample_uref_traj(args)

    ego = EgoVehicle(xref0, xrefN, pdf0, t_vec, system, env, args)

    opt = MPOptimizer(ego, args)

    ### 1. search valid trajectories
    #opt.search_good_traj()
    # with open(args.path_traj0 + "valid_traj", "rb") as f:
    #     uref_traj = pickle.load(f)
    # u_params = uref_traj[:, :, ::100]

    ### 2. optimize valid trajectories without considering the density
    u_params = torch.tensor([[[-1.0360, -0.0408, -0.0418, -0.0420, -0.0422, -0.0425,  1.9573,
          -0.0381, -0.0294, -0.0114],
         [-1.9572, -0.9625,  0.0329,  0.0284,  0.0240,  0.0196,  1.0143,
           0.0094,  2.0056,  1.0018]]]) # leads to collision
    u_params_opt = opt.optimize_traj(u_params, with_density=False)

    ### 3. optimize best trajectories with density predictions
    u_params_best = opt.optimize_traj(u_params_opt, with_density=True)






    print("end")