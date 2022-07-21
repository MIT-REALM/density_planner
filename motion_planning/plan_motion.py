from motion_planning.utils import sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from plots.plot_functions import plot_grid
from systems.sytem_CAR import Car
import pickle
import os
import logging
import sys
from motion_planning.optimization_objects import MotionPlanner
import numpy as np


if __name__ == '__main__':
    args = hyperparams.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device ="cpu"  #"cuda" if torch.cuda.is_available() else

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # 0. create environment
    objects = ["obstacleBottom", "pedLR", "pedRL", "bikerBT"]
    env = create_environment(args, timestep=100)
    plot_grid(env, args, timestep=0, save=False)
    plot_grid(env, args, timestep=20, save=False)
    plot_grid(env, args, timestep=40, save=False)
    plot_grid(env, args, timestep=60, save=False)
    plot_grid(env, args, timestep=80, save=False)
    plot_grid(env, args, save=False)

    xref0 = torch.tensor([0, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
    xrefN = torch.tensor([0, 8, 4, 1, 0]).reshape(1, -1, 1) #, [10, 5, 10, 6]]).reshape(2, -1, 1)

    ego = EgoVehicle(xref0, xrefN, env, args)
    opt = MotionPlanner(ego)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(opt.path_log + '/logfile.txt'),
                logging.StreamHandler(sys.stdout)])
    logging.info("Starting motion planning %s" % args.mp_name)

    ### 1. get initial trajectory
    if args.mp_find_traj:
        logging.info("Optimizing %d Random Trajectories" % args.mp_numtraj)
        opt.find_traj()
        logging.info("Optimization finished")
        u_params, cost_min = opt.find_best()
        logging.info("Best Trajectory with cost %.4f:" % cost_min)
        logging.info(u_params)
        u_params = u_params.detach()
    # best of 7
    elif args.input_type == "discr10":
        logging.info("Loading good parameters for input type discr10")
        u_params = torch.tensor([[[0.7080, -0.2266, 0.1678, -0.5573, -0.0683, -0.3008, 0.2324,
                -0.2033, -0.0517, 0.2307],
                [-0.1680, -0.2446, -0.3030, 0.8033, -0.3640, 0.1204, 0.4263,
                0.1605, -0.3789, 0.6836]]]) # cost=0.0094
    elif args.input_type == "discr5":
        logging.info("Loading good parameters for input type discr5")
        u_params = torch.tensor([[[ 0.0562,  0.7262, -0.7681, -0.0185, -0.2678],
         [-1.4503,  0.6820,  1.4877,  0.4498, -0.0598]]]) # cost=0.0128

    ### 2. optimize best trajectories with density predictions
    logging.info("Improving input parameters with density prediction")
    u_params, cost = opt.improve_traj(u_params)
    logging.info("Optimization finished with cost %.4f" % cost)
    logging.info(u_params)

    logging.info("Finished")

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


