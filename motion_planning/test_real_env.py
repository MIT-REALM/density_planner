from motion_planning.utils import sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, \
    create_turnR, create_pedLR
from env.environment import Environment as Env
import hyperparams
import torch
from plots.plot_functions import plot_grid
from systems.sytem_CAR import Car
import pickle
import os
import logging
import sys
from MotionPlannerGrad import MotionPlannerGrad
from MotionPlannerNLP import MotionPlannerNLP, MotionPlannerMPC
# import MotionPlannerNLP
import numpy as np

if __name__ == '__main__':
    args = hyperparams.parse_args()
    args.device = "cpu"

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    plot = True
    path_log = None
    num_initial = 10  # number of different initial state which will be evaluated

    ### loop through different environments
    for k in range(1):
        results = {}
        results["grad"] = {"time": [], "cost": [], "u": []}

        ### create environment and motion planning problem
        env = Env(args, init_time=0, end_time=12)

        # Compute grid from trajectory data
        env.run()
        env.grid = torch.transpose(env.grid, 0, 1) #@Andres: include in env.run()
        args.grid_size = [args.grid_size[1], args.grid_size[0]] #@Andres: include in env.run()

        plot_grid(env, args, timestep=0, save=False)
        plot_grid(env, args, timestep=20, save=False)
        plot_grid(env, args, timestep=40, save=False)
        plot_grid(env, args, timestep=60, save=False)
        plot_grid(env, args, timestep=80, save=False)
        plot_grid(env, args, timestep=100, save=False)

        # Create random initial state
        wpts_0 = env.generate_random_waypoint(0)
        wpts_N = env.generate_random_waypoint(12)

        xref0 = torch.tensor([wpts_0[0], wpts_0[1], 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        xrefN = torch.tensor([wpts_N[0], wpts_N[1], 4, 1, 0]).reshape(1, -1, 1)
        ego = EgoVehicle(xref0, xrefN, env, args)
        ego.system.X_MIN_MP[0, 0, 0] = args.environment_size[0] + 0.1
        ego.system.X_MIN_MP[0, 0, 0] = args.environment_size[2] + 0.1
        ego.system.X_MAX_MP[0, 0, 0] = args.environment_size[1] - 0.1
        ego.system.X_MAX_MP[0, 0, 0] = args.environment_size[3] - 0.1

        ### plan motion with density planner
        planner_grad = MotionPlannerGrad(ego, name="grad%d" % k, plot=plot, path_log=path_log)
        if k == 0:
            path_log = planner_grad.path_log

        up_grad, cost_grad, time_grad = planner_grad.plan_motion()
        results["grad"]["u"].append(up_grad)
        results["grad"]["time"].append(time_grad)

        ### compare cost from different initial states
        for j in range(num_initial):
            if j == 5:
                xe0 = torch.zeros(1, ego.system.DIM_X, 1)
            else:
                xe0 = ego.system.sample_xe0(1)
                xe0[:, 4, :] = 0

            ### evaluate trajectory for planner_grad
            cost = planner_grad.validate_traj(up_grad, xe0=xe0)
            results["grad"]["cost"].append(cost)

    print("end")



