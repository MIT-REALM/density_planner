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
from MotionPlannerGrad import MotionPlannerGrad
from MotionPlannerNLP import MotionPlannerNLP, MotionPlannerMPC
#import MotionPlannerNLP
import numpy as np


if __name__ == '__main__':
    args = hyperparams.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device ="cpu"  #"cuda" if torch.cuda.is_available() else

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    plot = True
    path_log = None
    num_initial = 10  # number of different initial state which will be evaluated

    ### loop through different environments
    for k in range(1):
        results= {}
        results["grad"] = {"time": [], "cost": [], "u": []}
        results["MPC"] = {"time": [], "cost": [], "u": []}
        results["NLP_up_cold"] = {"time": [], "cost": [], "u": []}
        results["NLP_up_warm"] = {"time": [], "cost": [], "u": []}
        results["NLP_utraj_cold"] = {"time": [], "cost": [], "u": []}
        results["NLP_utraj_warm"] = {"time": [], "cost": [], "u": []}

        ### create environment and motion planning problem
        env = create_environment(args, timestep=100) # @Andres: replace / adapt to use real data
        # plot_grid(env, args, timestep=0, save=False)
        # plot_grid(env, args, timestep=20, save=False)
        # plot_grid(env, args, timestep=40, save=False)
        # plot_grid(env, args, timestep=60, save=False)
        # plot_grid(env, args, timestep=80, save=False)
        # plot_grid(env, args, save=False)

        xref0 = torch.tensor([0, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        xrefN = torch.tensor([0., 8, 4, 1, 0]).reshape(1, -1, 1)
        ego = EgoVehicle(xref0, xrefN, env, args)

        ### plan motion with density planner
        planner_grad = MotionPlannerGrad(ego, name="grad%d" % k, plot=plot, path_log=path_log)
        if k == 0:
            path_log = planner_grad.path_log

        # up_grad, cost_grad, time_grad = planner_grad.plan_motion()
        up_grad = torch.Tensor([[[0.9255, -0.6596, -0.3638, 0.1527, 0.2528, -0.1136, -0.2866,
                  0.0067, -0.1316, -0.3437],
                 [0.2499, 0.4296, -0.1355, -0.0274, 0.6027, 0.3089, -0.5549,
                  0.0684, -0.0828, 0.2266]]]) #for seed 0
        # up_grad = torch.Tensor([[[-0.4295,  0.4854,  0.3976,  0.1772, -0.3519, -0.5208, -0.0646,
        #            0.8481, -0.5238, -0.7427],
        #          [-0.2112,  0.3185,  0.3122,  0.5391, -0.0937, -0.1417,  0.7684,
        #           -0.2717,  0.6058, -0.4975]]]) # for seed 1

        results["grad"]["u"].append(up_grad)
        #results["grad"]["time"].append(time_grad)

        ### compare with other motion planners from different initial states
        for j in range(num_initial):
            if j == 5:
                xe0 = torch.zeros(1, ego.system.DIM_X, 1)
            else:
                xe0 = ego.system.sample_xe0(1)
                xe0[:, 4, :] = 0

            ### evaluate trajectory for planner_grad
            cost = planner_grad.validate_traj(up_grad, xe0=xe0)
            results["grad"]["cost"].append(cost)

            ### compute trajectory with MPC
            planner_MPC = MotionPlannerMPC(ego, xe0=xe0, name="MPC%d.%d" % (k, j), plot=plot, path_log=path_log)
            u, cost, time = planner_MPC.plan_motion()
            results["MPC"]["u"].append(u)
            results["MPC"]["cost"].append(cost)
            results["MPC"]["time"].append(time)

            ### compute trajectory with MPC
            planner_MPC = MotionPlannerMPC(ego, xe0=xe0, name="MPC%d.%d" % (k, j), plot=plot, path_log=path_log, N_MPC=20)
            u, cost, time = planner_MPC.plan_motion()
            results["MPC"]["u"].append(u)
            results["MPC"]["cost"].append(cost)
            results["MPC"]["time"].append(time)

            ### compute input parameters with NLP
            planner_oracle = MotionPlannerNLP(ego, xe0=xe0, name="Oracle%d.%d" % (k, j), u0=None, plot=plot, path_log=path_log)
            u, cost, time = planner_oracle.plan_motion()
            results["NLP_up_cold"]["u"].append(u)
            results["NLP_up_cold"]["cost"].append(cost)
            results["NLP_up_cold"]["time"].append(time)

            ### compute input parameters with NLP with warm start
            planner_oracle = MotionPlannerNLP(ego, xe0=xe0, name="OracleWarm%d.%d" % (k, j), u0=up_grad, plot=plot, path_log=path_log)
            u, cost, time = planner_oracle.plan_motion()
            results["NLP_up_warm"]["u"].append(u)
            results["NLP_up_warm"]["cost"].append(cost)
            results["NLP_up_warm"]["time"].append(time)

            ### compute the whole input trajectory
            planner_oracle = MotionPlannerNLP(ego, xe0=xe0, name="OracleUtraj%d.%d" % (k, j), use_up=False, plot=plot, path_log=path_log)
            u, cost, time = planner_oracle.plan_motion()
            results["NLP_utraj_cold"]["u"].append(u)
            results["NLP_utraj_cold"]["cost"].append(cost)
            results["NLP_utraj_cold"]["time"].append(time)

            ### compute the whole input trajectory with warm start
            planner_oracle = MotionPlannerNLP(ego, xe0=xe0, u0=up_grad, name="OracleUtrajWarm%d.%d" % (k, j), use_up=False, plot=plot, path_log=path_log)
            u, cost, time = planner_oracle.plan_motion()
            results["NLP_utraj_warm"]["u"].append(u)
            results["NLP_utraj_warm"]["cost"].append(cost)
            results["NLP_utraj_warm"]["time"].append(time)


        # planner_sampling = MotionPlannerSampling(ego, plot=plot)
        # up_search, cost_search = planner_sampling.plan_motion()
        #
        # planner_search = MotionPlannerSearch(ego, plot=plot)
        # up_search, cost_search = planner_search.plan_motion()

        with open(path_log + "results%d" % k, "wb") as f:
            pickle.dump(results, f)
        for key_method in results.keys():
            print("#### Method: %s" % key_method)
            print("Average time: %.2f" % np.array(results[key_method]["time"]).mean())
            print("Failure rate: %.2f" % (np.array(results[key_method]["up"]) == None).sum() / num_initial)
    print("end")




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


