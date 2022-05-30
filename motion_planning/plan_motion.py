from motion_planning.utils import sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle, MPOptimizer
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from systems.sytem_CAR import Car
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
    #plot_grid(env, args)

    system = Car()
    pdf0 = sample_pdf(system, 5)
    xref0 = torch.tensor([0, -25, 1.5, 3]).reshape(1, -1, 1).type(torch.FloatTensor)
    xrefN = torch.tensor([0, 0, 4, 1]).reshape(1, -1, 1) #, [10, 5, 10, 6]]).reshape(2, -1, 1)
    t_vec = torch.arange(0, args.dt_sim * args.N_sim - 0.001, args.dt_sim * args.factor_pred)
    _, u_params = system.sample_uref_traj(args)

    ego = EgoVehicle(xref0, xrefN, pdf0, t_vec, system, env, args)

    opt = MPOptimizer(ego, args)
    opt.search_good_traj()

    cost = ego.predict_cost(args, u_params)

    ego.set_start_grid(args)
    plot_grid(ego, args, timestep=0)


    # 1. get initial input trajectory

    # 2. loop until cost small:
    #       a) compute density
    #       b) evaluate cost function and constraints
    #       c) optimize input




    print("end")