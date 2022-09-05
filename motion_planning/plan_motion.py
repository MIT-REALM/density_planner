import hyperparams
import torch
import pickle
import os
import logging
import sys
from motion_planning.example_objects import create_mp_task
from motion_planning.utils import make_path, initialize_logging, get_cost_table, get_cost_increase
from motion_planning.MotionPlannerGrad import MotionPlannerGrad, MotionPlannerSearch, MotionPlannerSampling
from motion_planning.MotionPlannerNLP import MotionPlannerNLP, MotionPlannerMPC
from plots.plot_functions import plot_cost, plot_traj
import numpy as np
import scipy.io


if __name__ == '__main__':
    args = hyperparams.parse_args()
    #args.mp_use_realEnv = True
    #args.mp_plot_envgrid = True
    args.random_seed = 10
    # args.mp_plot = True
    #args.mp_plot_traj = True
    stationary = True
    args.mp_load_old_opt = True

    ### choose methods
    opt_methods = ["grad", "search", "sampl"]#["sampl"] #["grad"] #
    mp_methods = ["grad", "MPC", "tube2MPC", "oracle"]
    #["grad", "grad_biased", "MPC", "MPC_biased", "tube2MPC", "tube2MPC_biased", "oracle"]
    #["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "tube2MPC", "tube2MPC_biased", "tube3MPC","tube3MPC_biased", "oracle"]
    ##
    #["tubeMPC", "tubeMPC_biased", "tube2MPC", "tube2MPC_biased", "tube3MPC", "tube3MPC_biased"]
    #["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "tube2MPC", "tube2MPC_biased", "tube3MPC","tube3MPC_biased", "oracle"]
    ##["grad", "grad_biased", "MPC", "MPC_biased", "tube2MPC", "tube2MPC_biased", "oracle"]
    #["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "tube2MPC", "tube2MPC_biased", "tube3MPC","tube3MPC_biased", "oracle"]

    ### settings
    filename_opt = "plots/motion/2022-09-04-07-30-12_statPlots_mpOpt_seed10-19/"
    filename_mp = "plots/motion/2022-09-04-07-30-12_statPlots_mpOpt_seed10-19/"
    #2022-09-03-09-16-27_compMP_tubeMPConeReturnTraj_seed41-49/" #2022-09-02-01-35-07_compMP_tubeMPConeReturnTraj_seed0-40/" #2022-09-01-10-56-42_compMP_seed40-49/"  #2022-09-01-22-50-55_test_seed0/" #2022-08-31-15-40-46_compMP_seed24-39/" #
    filename_mp2 = "plots/motion/2022-09-03-18-23-57_compMP_realEnv26_seed0-9/"
    filename_mp3 = "plots/motion/2022-09-04-04-08-51_compMP_realEnv8_seed7-10/"

    tube_size = 0.3
    tube2_size = 0.5
    tube3_size = 1

    ### generate plots
    plot = args.mp_plot
    plot_cost = args.mp_plot_cost
    plot_envgrid = args.mp_plot_envgrid
    plot_final = args.mp_plot_final
    save_results = args.mp_save_results
    load_old_opt = args.mp_load_old_opt
    load_old_mp = args.mp_load_old_mp
    video = args.mp_video
    use_realEnv = args.mp_use_realEnv

    path_log = initialize_logging(args, "test_realEnv_seed%d" % args.random_seed)

    if save_results:
        if load_old_opt:
            with open(filename_opt + "opt_results", "rb") as f:
                opt_results = pickle.load(f)
        else:
            opt_results = {}
            for opt_method in opt_methods:
                opt_results[opt_method] = {"time": [], "cost": [], "u": [], "x_traj": [], "cost_coll": 0, "cost_goal": 0, "cost_bounds": 0,
                                       "cost_uref": 0, "sum_time": 0, "num_valid": 0}
        if load_old_mp:
            with open(filename_mp + "mp_results", "rb") as f:
                mp_results = pickle.load(f)
            # with open(filename_mp2 + "mp_results", "rb") as f:
            #     mp_results2 = pickle.load(f)
            # for key, val in mp_results2.items():
            #     for key2 in val.keys():
            #         mp_results[key][key2] += mp_results2[key][key2]
            # mp_results = get_cost_increase(mp_methods, mp_results, 1000, 1000)
            # get_cost_table(mp_methods, mp_results)
            # mp_results = get_cost_increase(mp_methods, mp_results)
        else:
            mp_results = {}
            for mp_method in mp_methods:
                mp_results[mp_method] = {"time": [], "cost": [], "u": [], "x_traj": [], "cost_coll": 0, "cost_goal": 0,
                                     "cost_bounds": 0, "cost_uref": 0, "sum_time": 0, "num_valid": 0}

    ### loop through different environments
    for k in range(args.mp_num_envs):

        seed = args.random_seed + k
        torch.manual_seed(seed)
        np.random.seed(seed)

        ### create environment and motion planning problem
        ego = create_mp_task(args, seed, stationary)
        plot_traj(ego, opt_results, opt_methods, args, folder=filename_mp, traj_idx=k)
        continue
        
        ### test optimization methods
        for opt_method in opt_methods:
            name = opt_method + "%d" % (seed)
            if opt_method == "grad":
                planner = MotionPlannerGrad(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final, plot_cost=plot_cost)
            elif opt_method == "search":
                planner = MotionPlannerSearch(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final)
            elif opt_method == "sampl":
                planner = MotionPlannerSampling(ego, name=name, plot=plot, path_log=path_log, plot_final=plot_final)
            up, cost, time = planner.plan_motion()

            if save_results:
                opt_results[opt_method]["u"].append(up)
                opt_results[opt_method]["cost"].append(cost)
                opt_results[opt_method]["time"].append(time)
                opt_results[opt_method]["x_traj"].append(planner.xref_traj)
                if cost is not None:
                    opt_results[opt_method]["num_valid"] += 1
                    opt_results[opt_method]["sum_time"] += time
            if opt_method == "grad":
                planner_grad = planner
                up_grad = up

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
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased, tube=tube_size)
                        elif "tube2MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased, tube=tube2_size)
                        elif "tube3MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased, tube=tube3_size)
                        elif "safeMPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased, safe=True)
                        elif "MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased)
                        elif "oracle" in mp_method:
                            planner = MotionPlannerNLP(ego, xe0=xe0.clone(), name=name, plot=plot, path_log=path_log,
                                                       plot_final=plot_final, biased=biased)
                        else:
                            continue
                        up, cost, time = planner.plan_motion()
                    if save_results:
                        mp_results[mp_method]["u"].append(up)
                        mp_results[mp_method]["cost"].append(cost)
                        mp_results[mp_method]["time"].append(time)
                        mp_results[mp_method]["x_traj"].append(planner.x_traj)
                        if cost is not None:
                            mp_results[mp_method]["num_valid"] += 1
                            mp_results[mp_method]["sum_time"] += time

        if save_results:
            with open(path_log + "opt_results", "wb") as f:
                pickle.dump(opt_results, f)
            with open(path_log + "mp_results", "wb") as f:
                pickle.dump(mp_results, f)
        if args.mp_plot_traj:
            plot_traj(ego, mp_results, mp_methods, args, folder=path_log)
            #plot_traj(ego, opt_results, opt_methods, args, folder=path_log)

    if save_results:
        if len(opt_methods) != 0:
            opt_results = get_cost_increase(opt_methods, opt_results)
            get_cost_table(opt_methods, opt_results)
        if len(mp_methods) != 0:
            mp_results = get_cost_increase(mp_methods, mp_results)#, thr_coll=100, thr_goal=100)
            get_cost_table(mp_methods, mp_results)

    print("end")

# with open(filename_mp2 + "mp_results", "rb") as f:
#     mp_results2 = pickle.load(f)
# for method in mp_methods:
#     mp_results2[method] = mp_results[method]
# for method in mp_methods:
#     for key in ["time", "cost", "u", "x_traj"]:
#         mp_results2[method][key] = mp_results2[method][key][:41]

# for method in mp_methods:
#     for key in ["time", "cost", "u", "x_traj"]:
#         mp_results[method][key] += mp_results2[method][key]
#         mp_results[method][key] += mp_results3[method][key]
#     mp_results[method]["num_valid"] += mp_results2[method]["num_valid"] + mp_results3[method]["num_valid"]


