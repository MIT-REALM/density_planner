from motion_planning.utils import sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle, MPOptimizer
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from plots.plot_functions import plot_grid
from systems.sytem_CAR import Car
import pickle
import logging
import sys


if __name__ == '__main__':
    args = hyperparams.parse_args()
    torch.manual_seed(args.random_seed)

    # 0. create environment
    env = create_environment(["street", "obstacleBottom", "pedLR", "pedRL", "bikerBT"], args)
    plot_grid(env, args)
    env.forward_occupancy(step_size=100)
    plot_grid(env, args, timestep=20)
    plot_grid(env, args, timestep=40)
    plot_grid(env, args, timestep=60)
    plot_grid(env, args, timestep=80)
    plot_grid(env, args)
    env.enlarge_shape()

    xref0 = torch.tensor([0, -25, 1.5, 3]).reshape(1, -1, 1).type(torch.FloatTensor)
    xrefN = torch.tensor([0, 0, 4, 1]).reshape(1, -1, 1) #, [10, 5, 10, 6]]).reshape(2, -1, 1)

    ego = EgoVehicle(xref0, xrefN, env, args)
    opt = MPOptimizer(ego)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(ego.path_log + '/logfile.txt'),
                logging.StreamHandler(sys.stdout)])
    logging.info("Starting motion planning %s" % args.mp_name)

    ### 1. get initial trajectory
    if args.mp_init == "search":
        opt.search_good_traj(num_traj=args.mp_search_numtraj)
        logging.info("search finished with %d valid reference trajectories" % len(opt.found_traj))
        u_params = opt.found_traj[0][0]
    elif args.mp_init == "search_best":
        u_params, cost = opt.find_best_traj(load=False)
        logging.info("best reference trajectory found with cost %.2f" % cost)
    elif args.mp_init == "load":
        with open(args.path_traj0 + "found_traj", "rb") as f:
            valid_traj = pickle.load(f)
        logging.info("%d valid reference trajectories loaded" % len(valid_traj))
        u_params = valid_traj[0][0]
    elif args.mp_init == "load_best":
        u_params, cost = opt.find_best_traj(load=True)
        logging.info("best reference trajectory found with cost %.2f" % cost)
    elif args.mp_init == "best":
        u_params = torch.tensor([[[0.5000, 0.0000, -0.5000, -0.5000, 0.5000],
             [0.0000, -0.5000, 0.5000, 0.5000, 0.0000]]]) #best found trajectory
    elif args.mp_init == "zeros":
        u_params = torch.zeros((1, 2, 5))
        logging.info("zeros reference trajectory")
    elif args.mp_init == "random":
        u_params = 0.5 * torch.randn((1, 2, 5))
        u_params = u_params.clamp(ego.system.UREF_MIN, ego.system.UREF_MAX)
        logging.info("random reference trajectory")
    logging.info(u_params)

    if args.mp_anim_initial:
        ego.animate_traj(u_params, name="initialTraj")
    else:
        uref_traj, xref_traj = ego.system.u_params2ref_traj(ego.xref0, u_params, args, short=True)
        ego.visualize_xref(xref_traj, save=True, show=False, name="initialTraj_%s" % args.mp_init, folder=ego.path_log)

    ### 2. optimize valid trajectories without considering the density
    if args.mp_opt_normal:
        u_params = opt.optimize_traj(u_params, with_density=False)
        logging.info("reference trajectory was optimized:")
        logging.info(u_params)
        #u_params_opt = torch.tensor([[[0.9989, -0.7407, -0.3733, -0.0974, -0.0172],
        #     [-0.2134, 0.1232, 0.3187, 0.2029, 0.0679]]], requires_grad=True) #optimized from zero traj, no collision

    # with open("plots/motion/2022-06-13-14-04-00_mp_random/2022-06-13-14-04-02_opt/results", "rb") as f:
    #     res = pickle.load(f)
    # u_params = res[0]
    # ego.animate_traj(u_params, name="finalTraj")

    ### 3. optimize best trajectories with density predictions
    if args.mp_opt_density:
        u_params = opt.optimize_traj(u_params, with_density=True)
        logging.info("trajectory was optimized with density estimates:")
        logging.info(u_params)

    if args.mp_anim_final:
        ego.animate_traj(u_params, name="finalTraj")
    else:
        uref_traj, xref_traj = ego.system.u_params2ref_traj(ego.xref0, u_params, args, short=True)
        ego.visualize_xref(xref_traj, save=True, show=False, name="finalTraj", folder=ego.path_log)

    logging.info("finished")

'''
gradients wrt u_params:
when u_params = torch.tensor([[[-1.0360, -0.0408, -0.0418, -0.0420, -0.0422, -0.0425,  1.9573,
          -0.0381, -0.0294, -0.0114],
         [-1.9572, -0.9625,  0.0329,  0.0284,  0.0240,  0.0196,  1.0143,
           0.0094,  2.0056,  1.0018]]])
           
cost_coll --> tensor([[[-1.3976, -1.4109, -1.4588, -1.4843, -1.5194, -1.5642, -1.5418,
          -0.9931, -0.2097,  0.0000],
         [ 3.6451,  3.0688,  2.5036,  1.9473,  1.4009,  0.8655,  0.3345,
           0.1061,  0.0185,  0.0000]]])
            --> weight to get grad < 0.1 = 1e-2
            --> factor for importance: 1...5
cost_goal --> tensor([[[ 103.9837,  128.7704,  135.6620,  138.9141,  143.0755,  148.0411,
           153.3208,  136.8134,  103.5397,   39.7553],
         [-189.7535, -168.2200, -151.3658, -135.7018, -121.2646, -108.0862,
           -89.4044,  -62.6490,  -37.9397,  -12.6351]]])
            --> weight to get grad < 0.1 = 1e-4
            --> factor for importance: 1
cost_uref --> tensor([[[-20.7200,  -0.8160,  -0.8360,  -0.8400,  -0.8440,  -0.8500,  39.1460,
           -0.7620,  -0.5880,  -0.2280],
         [-39.1440, -19.2500,   0.6580,   0.5680,   0.4800,   0.3920,  20.2860,
            0.1880,  40.1120,  20.0360]]])           
            --> weight to get grad < 0.1 = 1e-3 
            --> factor for importance: 1e-1
cost_bounds --> tensor([[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000],
         [-0.8166, -0.8166, -0.5443, -0.1355, -0.0011,  0.0000,  0.0000,
           0.0000,  0.0000,  0.0000]]])           
            --> weight to get grad < 0.1 = 1e-1 
            --> factor for importance: 10
cost_obs_dist --> tensor([[[-30.6858,  -5.5381,  -1.3426,  -4.6947,  -2.4206],
         [ -1.9916,  -6.3574,  -6.8133,  -0.6762,   0.1909]]])
            --> weight: 1e-3
'''


