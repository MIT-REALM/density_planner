import hyperparams
import pickle
from plots.plot_functions import plot_traj


"""
script to evaluate the optimization method and to compare density planner with baseline methods
"""
if __name__ == '__main__':
    ### load hyperparameters
    args = hyperparams.parse_args()

    # ablation study of the optimization method (compare gradient-based, search-based and sampling-based method)
    opt_methods = ["grad", "search", "sampl"] #
    mp_methods =["grad", "MPC", "tube2MPC", "oracle"]  # []  #

    # create logging folder
    #filename = "2022-09-16-14-07-07_compMP_animation_seed0-7" #
    filename = "2022-09-16-13-08-10_compOpt_animation_seed0-4"
    #filename = "2022-09-16-16-51-45_compMP_env8_animation_seed0-4"

    for k in [0]:
        traj_idx = k

        name_ego = "ego%d" % k
        path_log = args.path_plot_motion + filename + "/"

        with open(path_log + "opt_results", "rb") as f:
            opt_results = pickle.load(f)
        with open(path_log + "mp_results", "rb") as f:
            mp_results = pickle.load(f)
        with open(path_log + name_ego, "rb") as f:
            ego_dict = pickle.load(f)

        #plot_traj(ego_dict, mp_results, mp_methods, args, traj_idx=traj_idx, folder=path_log, animate=True, include_density=False)
        #plot_traj(ego_dict, mp_results, mp_methods, ego_dict["args"], traj_idx=traj_idx, folder=path_log, animate=True, include_density=False)
        plot_traj(ego_dict, opt_results, opt_methods, args, traj_idx=traj_idx, folder=path_log, animate=True, include_density=True)



