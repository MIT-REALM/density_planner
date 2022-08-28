from motion_planning.utils import bounds2array, sample_pdf, pred2grid
from motion_planning.simulation_objects import Environment, EgoVehicle
from motion_planning.example_objects import create_environment, create_crossing4w, create_pedRL, create_street, create_turnR, create_pedLR
import hyperparams
import torch
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import pickle
import os
import logging
import sys
from motion_planning.utils import make_path, initialize_logging
from motion_planning.MotionPlannerGrad import MotionPlannerGrad, MotionPlannerSearch, MotionPlannerSampling
from motion_planning.MotionPlannerNLP import MotionPlannerNLP, MotionPlannerMPC
import numpy as np
import scipy.io


if __name__ == '__main__':
    args = hyperparams.parse_args()
    args.device = "cpu"


    ### generate plots
    plot = False
    plot_envgrid = False
    plot_final = False
    save_results = True
    load_old = False
    video = False
    use_realEnv = True
    
    ### choose methods
    opt_methods = ["grad"] #["grad", "search", "sampl"]#["sampl"] #
    mp_methods = ["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "oracle"] #["tubeMPC"] #[]

    ### settings
    biased = True
    tube_size = 0.5
    path_log = initialize_logging(args, "testMPC_seed%d" % args.random_seed)

    if save_results:
        if load_old:
            with open("plots/motion/2022-08-27-18-30-19_compOpt_server_seed0-19/" + "opt_results", "rb") as f:
                opt_results = pickle.load(f)
        else:
            opt_results = {}
        for opt_method in opt_methods:
            opt_results[opt_method] = {"time": [], "cost": [], "u": [], "cost_coll": 0, "cost_goal": 0, "cost_bounds": 0,
                                       "cost_uref": 0, "sum_time": 0, "num_valid": 0}
        mp_results = {}
        for mp_method in mp_methods:
            mp_results[mp_method] = {"time": [], "cost": [], "u": [], "cost_coll": 0, "cost_goal": 0, "cost_bounds": 0,
                                     "cost_uref": 0, "sum_time": 0, "num_valid": 0}

    ### loop through different environments
    init_time = 0
    for k in range(args.mp_num_envs):

        seed = args.random_seed + k
        torch.manual_seed(seed)
        np.random.seed(seed)

        ### create environment and motion planning problem
        if use_realEnv:
            grid_sum = 0
            ### create environment and motion planning problem
            while grid_sum < 11e5:
                env = Env(args, init_time=init_time, end_time=init_time + 11)
                env.run()
                init_time += np.random.randint(12, 20)
                grid_sum = env.grid[env.grid != 1].sum()
        else:
            env = create_environment(args, timestep=100)
        #scipy.io.savemat(args.path_matlab + "env%d.mat" % k, mdict={'arr': np.array(env.grid)})
        #array_bounds = bounds2array(env, args)
        #scipy.io.savemat(args.path_matlab + "bounds%d.mat" % k, mdict={'arr': array_bounds})

        if plot_envgrid:
            plot_grid(env, args, timestep=0, save=False)
            plot_grid(env, args, timestep=20, save=False)
            plot_grid(env, args, timestep=40, save=False)
            plot_grid(env, args, timestep=60, save=False)
            plot_grid(env, args, timestep=80, save=False)
            plot_grid(env, args, save=False)

        xref0 = torch.tensor([0, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        xrefN = torch.tensor([0., 8, 4, 1, 0]).reshape(1, -1, 1)
        ego = EgoVehicle(xref0, xrefN, env, args, video=video)
        
        ### test optimization methods
        for opt_method in opt_methods:
            name = opt_method + "%d" % (seed)
            if opt_method == "grad":
                planner = MotionPlannerGrad(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final)
            elif opt_method == "search":
                planner = MotionPlannerSearch(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final)
            elif opt_method == "sampl":
                planner = MotionPlannerSampling(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final)
            up, cost, time = planner.plan_motion()

            if save_results:
                opt_results[opt_method]["u"].append(up)
                opt_results[opt_method]["cost"].append(cost)
                opt_results[opt_method]["time"].append(time)
                if cost is not None:
                    opt_results[opt_method]["num_valid"] += 1
                    opt_results[opt_method]["sum_time"] += time
            if opt_method == "grad":
                planner_grad = planner
                up_grad = up

        if save_results and len(opt_methods) != 0:
            for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref"]:
                cost_min = np.inf
                for opt_method in opt_methods:
                    if opt_results[opt_method]["cost"][-1] is not None and opt_results[opt_method]["cost"][-1][s] < cost_min:
                        cost_min = opt_results[opt_method]["cost"][-1][s]
                for opt_method in opt_methods:
                    if opt_results[opt_method]["cost"][-1] is not None:
                        opt_results[opt_method][s] += opt_results[opt_method]["cost"][-1][s] - cost_min

        ### compare with other motion planners from different initial states
        if len(mp_methods) != 0:
            for j in range(args.mp_num_initial):
                # if j == 1:
                #     xe0 = torch.zeros(1, ego.system.DIM_X, 1)
                # else:
                xe0 = ego.system.sample_xe0(1)
                # if not biased:
                #     xe0[:, 4, :] = 0

                for mp_method in mp_methods:
                    name = mp_method + "%d.%d" % (seed, j)
                    if "biased" in mp_method:
                        biased = True
                    else:
                        biased = False

                    if "grad"in mp_method:
                        cost, time = planner_grad.validate_traj(up_grad, xe0=xe0.clone(), return_time=True, biased=biased)
                        up = up_grad
                    else:
                        if "tubeMPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log, plot_final=plot_final, biased=biased, tube=tube_size)
                        elif "safeMPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log, plot_final=plot_final, biased=biased, safe=True)
                        elif "MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log, plot_final=plot_final, biased=biased)
                        elif "oracle" in mp_method:
                            planner = MotionPlannerNLP(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log, plot_final=plot_final, biased=biased)
                        else:
                            continue
                        up, cost, time = planner.plan_motion()
                    if save_results:
                        mp_results[mp_method]["u"].append(up)
                        mp_results[mp_method]["cost"].append(cost)
                        mp_results[mp_method]["time"].append(time)
                        if cost is not None:
                            mp_results[mp_method]["num_valid"] += 1
                            mp_results[mp_method]["sum_time"] += time

                if save_results:
                    for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref"]:
                        cost_min = np.inf
                        for mp_method in mp_methods:
                            if mp_results[mp_method]["cost"][-1] is not None and \
                                    mp_results[mp_method]["cost"][-1][s] < cost_min:
                                cost_min = mp_results[mp_method]["cost"][-1][s]
                        for mp_method in mp_methods:
                            if mp_results[mp_method]["cost"][-1] is not None:
                                mp_results[mp_method][s] += mp_results[mp_method]["cost"][-1][s] - cost_min


    if save_results:
        with open(path_log + "opt_results", "wb") as f:
            pickle.dump(opt_results, f)
        with open(path_log + "mp_results", "wb") as f:
            pickle.dump(mp_results, f)
    if save_results:
        if len(opt_methods) != 0:
            for opt_method in opt_methods:
                logging.info("%s: number valid: %d" % (opt_method, opt_results[opt_method]["num_valid"]))
                for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "sum_time"]:
                    logging.info("%s: %s: %.2f" % (opt_method, s, opt_results[opt_method][s] / opt_results[opt_method]["num_valid"]))

            print("#### TABLE:")
            for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "sum_time"]:
                print("#### %s:" % s)
                for l in range(args.mp_num_envs):
                    print("%d" % l, end=" & ")
                    for opt_method in opt_methods:
                        if opt_results[opt_method]["cost"][l] is not None:
                            if s == "sum_time":
                                print("%.3f" % opt_results[opt_method]["time"][l], end=" & ")
                            else:
                                print("%.3f" % opt_results[opt_method]["cost"][l][s].item(), end=" & ")
                        else:
                            print(" - ", end=" & ")
                    print(" \\ ")
        if len(mp_methods) != 0:
            for mp_method in mp_methods:
                logging.info("%s: number valid: %d" % (mp_method, mp_results[mp_method]["num_valid"]))
                for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "sum_time"]:
                    logging.info("%s: %s: %.2f" % (mp_method, s, mp_results[mp_method][s] / mp_results[mp_method]["num_valid"]))
            print("#### TABLE:")
            for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "sum_time"]:
                print("#### %s:" % s)
                for l in range(args.mp_num_envs):
                    print("%d" % l, end=" & ")
                    for mp_method in mp_methods:
                        if mp_results[mp_method]["cost"][l] is not None:
                            if s == "sum_time":
                                print("%.3f" % mp_results[mp_method]["time"][l], end=" & ")
                            else:
                                print("%.3f" % mp_results[mp_method]["cost"][l][s].item(), end=" & ")
                        else:
                            print(" - ", end=" & ")
                    print(" \\ ")

    print("end")


