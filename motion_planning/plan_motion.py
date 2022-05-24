from motion_planning.utils import plot_grid, sample_pdf, pdf2grid
from motion_planning.simulation_objects import Environment, EgoVehicle
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from systems.sytem_CAR import Car
from systems.utils import get_mesh_pos


if __name__ == '__main__':
    args = hyperparams.parse_args()
    torch.manual_seed(args.random_seed)

    # 0. create environment
    # env = create_environment(["street", "obstacleBottom", "pedLR"], args)
    # plot_grid(env, args)
    # env.forward_occupancy(step_size=50)
    # plot_grid(env, args)

    system = Car()
    g = get_mesh_pos([6, 5, 3, 2])
    start_pdf = sample_pdf(system, 5)
    # grid = pdf2grid(start_pdf, system.XE0_MIN.flatten(), system.XE0_MAX.flatten(), args)
    #
    # xe0 = torch.rand(50, system.DIM_X, 1) * (system.XE0_MAX - system.XE0_MIN) + system.XE0_MIN
    # probs = start_pdf(xe0)

    ego = EgoVehicle(system, args, start=[0,-25], goal=[-5, 5, -5, 5], pdf0=start_pdf)
    ego.set_start_grid(args)
    plot_grid(ego, args, timestep=0)

    # 1. get initial input trajectory

    # 2. loop until cost small:
    #       a) compute density
    #       b) evaluate cost function and constraints
    #       c) optimize input




    print("end")