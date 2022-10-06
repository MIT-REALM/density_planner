import hyperparams
import torch
import pickle
from motion_planning.example_objects import create_mp_task
from motion_planning.utils import initialize_logging, get_cost_table, get_cost_increase
from motion_planning.MotionPlannerGrad import MotionPlannerGrad, MotionPlannerSearch, MotionPlannerSampling
from motion_planning.MotionPlannerNLP import MotionPlannerNLP, MotionPlannerMPC
from plots.plot_functions import plot_traj
import numpy as np


if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    ### choose methods
    if args.mp_setting == "ablation":
        # ablation study of the optimization method (compare gradient-based, search-based and sampling-based method)
        opt_methods = ["grad", "search", "sampl"]
        mp_methods = []
    elif args.mp_setting == "artificial":
        # comparison of all motion planning approaches in the environments generated from artificial data
        opt_methods = ["grad"]
        mp_methods = ["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "tube2MPC",
                      "tube2MPC_biased", "tube3MPC", "tube3MPC_biased", "oracle"]
    elif args.mp_setting == "real":
        # comparison of the motion planning approaches in the environments generated from real-world data
        opt_methods = ["grad"]
        mp_methods = ["grad", "grad_biased", "MPC", "MPC_biased", "tube2MPC", "tube2MPC_biased", "oracle"]
        args.mp_real_env = True
    else:  # custom configuration, adapt to your case
        # choose optimization methods from ["grad", "search", "sampl"]
        opt_methods = ["grad"]
        # choose motion planning methods from  ["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased",
        #                                       "tube2MPC", "tube2MPC_biased", "tube3MPC","tube3MPC_biased", "oracle"]
        mp_methods = ["grad"] #, "MPC", "tube2MPC", "oracle"]
        # mp_methods = ["grad", "grad_biased", "MPC", "MPC_biased", "tubeMPC", "tubeMPC_biased", "tube2MPC",
        #               "tube2MPC_biased", "tube3MPC", "tube3MPC_biased", "oracle"]

    ### settings
    # specify tube radius of tube-based MPC
    tube_size = 0.3  # radius of "tubeMPC"
    tube2_size = 0.5  # radius of "tube2MPC"
    tube3_size = 1  # radius of "tube3MPC"

    # create logging folder
    path_log = initialize_logging(args, args.mp_name + "_startingSeed%d" % args.random_seed)

    # create variables for saving the results
    if args.mp_save_results:
        # optimization results:
        if args.mp_load_old_opt:
            with open(args.mp_filename_opt + "opt_results", "rb") as f:
                opt_results = pickle.load(f)
        else:
            opt_results = {}
            for opt_method in opt_methods:
                opt_results[opt_method] = {"time": [], "cost": [], "u": [], "x_trajs": [], "x_traj": [],
                                           "rho_traj": [], "cost_coll": 0, "cost_goal": 0, "cost_bounds": 0,
                                           "cost_uref": 0, "sum_time": 0, "num_valid": 0}

        # motion planning results:
        if args.mp_load_old_mp:
            with open(args.mp_filename_mp + "mp_results", "rb") as f:
                mp_results = pickle.load(f)
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
        ego = create_mp_task(args, seed)
        
        ### test optimization methods
        for opt_method in opt_methods:
            name = opt_method + "%d" % (seed)
            if opt_method == "grad":
                planner = MotionPlannerGrad(ego, name=name, path_log=path_log)
            elif opt_method == "search":
                planner = MotionPlannerSearch(ego, name=name, path_log=path_log)
            elif opt_method == "sampl":
                planner = MotionPlannerSampling(ego, name=name, path_log=path_log)
            up, cost, time = planner.plan_motion()

            if args.mp_save_results:
                opt_results[opt_method]["u"].append(up)
                opt_results[opt_method]["cost"].append(cost)
                opt_results[opt_method]["time"].append(time)
                opt_results[opt_method]["x_traj"].append(planner.xref_traj)
                opt_results[opt_method]["x_trajs"].append(planner.x_traj)
                opt_results[opt_method]["rho_traj"].append(planner.rho_traj)
                if cost is not None:
                    opt_results[opt_method]["num_valid"] += 1
                    opt_results[opt_method]["sum_time"] += time
            if opt_method == "grad":
                planner_grad = planner
                up_grad = up


        if len(mp_methods) != 0:
            ### compare with other motion planners starting from different initial states
            for j in range(args.mp_num_initial):
                # sample initial state from initial density distribution
                xe0 = ego.system.sample_xe0(1)

                for mp_method in mp_methods:
                    name = mp_method + "%d.%d" % (seed, j)
                    if "biased" in mp_method:
                        biased = True
                    else:
                        biased = False

                    # run motion planner
                    if "grad"in mp_method:
                        cost, time = planner_grad.validate_traj(up_grad, xe0=xe0.clone(), return_time=True, biased=biased)
                        up = up_grad
                    else:
                        if "tubeMPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased, tube=tube_size)
                        elif "tube2MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased, tube=tube2_size)
                        elif "tube3MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased, tube=tube3_size)
                        elif "safeMPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased, safe=True)
                        elif "MPC" in mp_method:
                            planner = MotionPlannerMPC(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased)
                        elif "oracle" in mp_method:
                            planner = MotionPlannerNLP(ego, xe0=xe0.clone(), name=name, path_log=path_log,
                                                       biased=biased)
                        else:
                            continue
                        up, cost, time = planner.plan_motion()
                    if args.mp_save_results:
                        mp_results[mp_method]["u"].append(up)
                        mp_results[mp_method]["cost"].append(cost)
                        mp_results[mp_method]["time"].append(time)
                        mp_results[mp_method]["x_traj"].append(planner.x_traj)
                        if cost is not None:
                            mp_results[mp_method]["num_valid"] += 1
                            mp_results[mp_method]["sum_time"] += time

        ego_dict = {"grid": ego.env.grid,
                    "start": ego.xref0,
                    "goal": ego.xrefN,
                    "args": args}
        if args.mp_save_results:
            with open(path_log + "opt_results", "wb") as f:
                pickle.dump(opt_results, f)
            with open(path_log + "mp_results", "wb") as f:
                pickle.dump(mp_results, f)
            with open(path_log + "ego%d" % seed, "wb") as f:
                pickle.dump(ego_dict, f)
        if args.mp_plot_traj:
            if len(mp_methods) != 0:
                plot_traj(ego_dict, mp_results, mp_methods, args, folder=path_log)
            if len(opt_methods) != 0:
                plot_traj(ego_dict, opt_results, opt_methods, args, traj_idx=k, folder=path_log, animate=True)

    if args.mp_save_results:
        if len(opt_methods) != 0:
            opt_results = get_cost_increase(opt_methods, opt_results)  # create table (see appendix)
            get_cost_table(opt_methods, opt_results)  # get criteria
        if len(mp_methods) != 0:
            mp_results = get_cost_increase(mp_methods, mp_results, 1000, 1000)
            get_cost_table(mp_methods, mp_results)  # create table (see appendix)
            mp_results = get_cost_increase(mp_methods, mp_results)  # get criteria
            get_cost_table(mp_methods, mp_results)

    print("end")


