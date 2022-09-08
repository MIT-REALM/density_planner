from motion_planning.simulation_objects import EgoVehicle
from env.environment import Environment as Env
import hyperparams
import torch
from motion_planning.MotionPlannerGrad import MotionPlannerGrad
import numpy as np
from plots.plot_functions import plot_grid

if __name__ == '__main__':
    args = hyperparams.parse_args()
    args.device = "cpu"

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    plot = False
    plot_final = True
    path_log = None
    num_initial = 10  # number of different initial state which will be evaluated
    init_time = 0

    ### loop through different environments
    for k in range(1):
        results = {}
        results["grad"] = {"time": [], "cost": [], "u": []}
        trajectory = {"method": [], "xref_traj": []}

        grid_sum = 0
        ### create environment and motion planning problem
        while grid_sum < 9.4e5:
            env = Env(args, init_time=init_time, end_time=init_time + 11)
            env.run()
            init_time += np.random.randint(12, 20)
            grid_sum = env.grid[env.grid != 1].sum()
            print(grid_sum)

        # Compute grid from trajectory data
        plot_grid(env, args, timestep=0, save=False)
        plot_grid(env, args, timestep=20, save=False)
        plot_grid(env, args, timestep=40, save=False)
        plot_grid(env, args, timestep=60, save=False)
        plot_grid(env, args, timestep=80, save=False)
        plot_grid(env, args, timestep=100, save=False)

        # Create random initial state
        dist_start_goal = 0
        pos_0 = env.generate_random_waypoint(time=init_time)
        while dist_start_goal < 10 or dist_start_goal > 70:
            # pos_0 = np.array([-15, -15]) + 10 * np.random.rand(2) #env.generate_random_waypoint(0)
            theta_0 = 1.6 * np.random.rand(1)
            v_0 = 1 + 8 * np.random.rand(1)
            # pos_N = np.array([25, 15]) + 10 * np.random.rand(2) #env.generate_random_waypoint(10)
            pos_N = env.generate_random_waypoint(time=init_time + 11)
            dist_start_goal = np.sqrt((pos_0[0]-pos_N[0])**2 + (pos_0[1]-pos_N[1])**2)
        print("start")
        print(pos_0)
        print("end")
        print(pos_N)
        xref0 = torch.tensor([pos_0[0], pos_0[1], theta_0[0], v_0[0], 0]).reshape(1, -1, 1).type(torch.FloatTensor)
        xrefN = torch.tensor([pos_N[0], pos_N[1], 0, 0, 0]).reshape(1, -1, 1)
        ego = EgoVehicle(xref0, xrefN, env, args)
        ego.system.X_MIN_MP[0, 0, 0] = args.environment_size[0] + 0.1
        ego.system.X_MIN_MP[0, 1, 0] = args.environment_size[2] + 0.1
        ego.system.X_MAX_MP[0, 0, 0] = args.environment_size[1] - 0.1
        ego.system.X_MAX_MP[0, 1, 0] = args.environment_size[3] - 0.1

        ### plan motion with density planner
        planner_grad = MotionPlannerGrad(ego, name="grad%d" % k, plot=plot, path_log=path_log, plot_final=plot_final)
        if k == 0:
            path_log = planner_grad.path_log

        up_grad, cost_grad, time_grad = planner_grad.plan_motion()
        results["grad"]["u"].append(up_grad)
        results["grad"]["time"].append(time_grad)
        results["grad"]["cost"].append(cost_grad)
        trajectory["method"].append("grad")
        trajectory["xref_traj"].append(planner_grad.xref_traj)
        # Create animation
        env.create_overlay_animation(xref0.numpy().reshape((5,)), xrefN.numpy().reshape((5,)), trajectory)


    print("end")



