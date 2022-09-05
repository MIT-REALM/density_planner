from motion_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np
from motion_planning.utils import pos2gridpos, find_start_goal, check_start_goal
from motion_planning.simulation_objects import EgoVehicle
from plots.plot_functions import plot_grid
from env.environment import Environment as Env
import torch
import logging


def create_mp_task(args, seed, stationary=False):
    logging.info("")
    logging.info("###################################################################")
    logging.info("###################################################################")
    if args.mp_use_realEnv:
        ### create environment and motion planning problem
        init_time = seed * 100
        valid = False
        while not valid:
            grid_sum = 0
            valid = True
            while grid_sum < args.mp_min_gridSum:
                env = Env(args, init_time=init_time, end_time=init_time + 11)
                init_time += 12
                env.run()
                grid_sum = env.grid[env.grid != 1].sum()
            if args.environment_size[1] > 50:
                args.environment_size[0] -= (args.environment_size[1] - 50)
                args.environment_size[1] = 50
            if args.environment_size[3] > 50:
                args.environment_size[2] -= (args.environment_size[3] - 50)
                args.environment_size[3] = 50
            if stationary:
                env.grid[:, :, :] = env.grid[:, :, [40]]
            logging.info("Loading Real Environment %d with spread %.2f, starting time = %.1f (seed %d)" % (
                                        env.config["dataset"]["recording"],
                                        env.config["grid"]["spread"], init_time - 12, seed))
            xref0, xrefN = find_start_goal(env, args)
            if xref0 is None:
                valid = False
    else:
        env = create_environment(args, timestep=100, stationary=stationary)
        logging.info("Loading Simulated Environment (seed %d)" % (seed))
        if seed < 20:
            xref0 = torch.tensor([0, -28, 1.5, 3, 0]).reshape(1, -1, 1).type(torch.FloatTensor)
            xrefN = torch.tensor([0., 8, 4, 1, 0]).reshape(1, -1, 1)
        else:
            valid = False
            while not valid:
                pos_0 = np.array([-5, 5]) + np.array([10, 5]) * np.random.rand(2)
                pos_N = np.array([-5, -30]) + np.array([10, 10]) * np.random.rand(2)
                theta_0 = -0.5 - 1.6 * np.random.rand(1)
                v_0 = 1 + 8 * np.random.rand(1)
                valid = check_start_goal(env.grid, pos_0, pos_N, theta_0, v_0, args)
            xref0 = torch.tensor([pos_0[0], pos_0[1], theta_0[0], v_0[0], 0]).reshape(1, -1, 1).type(torch.FloatTensor)
            xrefN = torch.tensor([pos_N[0], pos_N[1], 0, 0, 0]).reshape(1, -1, 1)
    if args.mp_plot_envgrid:
        for t in [1, 20, 40, 60, 80, 100]:
            plot_grid(env, args, timestep=t, save=False)
    logging.info("Start State: [%.1f, %.1f, %.1f, %.1f]" % (xref0[0, 0, 0], xref0[0, 1, 0], xref0[0, 2, 0], xref0[0, 3, 0]))
    logging.info("Goal Position: [%.1f, %.1f]" % (xrefN[0, 0, 0], xrefN[0, 1, 0]))
    # scipy.io.savemat(args.path_matlab + "env%d.mat" % k, mdict={'arr': np.array(env.grid)})
    # array_bounds = bounds2array(env, args)
    # scipy.io.savemat(args.path_matlab + "bounds%d.mat" % k, mdict={'arr': array_bounds})

    ego = EgoVehicle(xref0, xrefN, env, args, video=args.mp_video)
    if args.mp_use_realEnv:
        ego.system.X_MIN_MP[0, 0, 0] = args.environment_size[0] + 0.1
        ego.system.X_MIN_MP[0, 1, 0] = args.environment_size[2] + 0.1
        ego.system.X_MAX_MP[0, 0, 0] = args.environment_size[1] - 0.1
        ego.system.X_MAX_MP[0, 1, 0] = args.environment_size[3] - 0.1
    return ego


def create_environment(args, object_str_list=None, name="environment", timestep=0, stationary=False):
    objects = create_street(args)
    if object_str_list is not None:
        for obj_str in object_str_list:
            obj = globals()["create_" + obj_str](args)
            if isinstance(obj, list):
                objects += obj
            else:
                objects.append(obj)
    else:
        num_static = np.random.randint(4, 8)
        num_dynamics = np.random.randint(6, 10)
        for i in range(num_static + num_dynamics):
            wide = np.random.randint(5, 40) / 10
            height = np.random.randint(5, 40) / 10
            xstart = np.random.randint(-7, 7)
            ystart = np.random.randint(-23, 5)
            certainty = np.random.randint(3, 10) / 10
            spread = np.random.randint(1, 30)
            obs = np.array([xstart, xstart + wide, ystart, ystart + height, certainty, spread])
            if not stationary and i >= num_static:
                vx = np.random.randint(-2, 2)
                vy = np.random.randint(-2, 2)
                obj = DynamicObstacle(args, name="staticObs%d" % (i - num_static), coord=obs, velocity_x=vx,
                                      velocity_y=vy)
            else:
                obj = StaticObstacle(args, name="staticObs%d" % i, coord=obs)
            objects.append(obj)

    environment = Environment(objects, args, name=name)
    if timestep > 0:
        environment.forward_occupancy(step_size=timestep)
    return environment

def create_street(args):
    static_obstacles = {
        "left": np.array([args.environment_size[0], -7, args.environment_size[2], args.environment_size[3], 1, 30]), #(x1, x2, y1, y2, certainty, spread)
        "right": np.array([7, args.environment_size[1], args.environment_size[2], args.environment_size[3], 1, 30])}
    objs = []
    for key, value in static_obstacles.items():
        objs.append(StaticObstacle(args, coord=value, name="street "+key, timestep=0))
    return objs

def create_turnR(args):
    static_obstacles = {
        "left": np.array([args.environment_size[0], -30, args.environment_size[2], args.environment_size[3], 1, 1]), #(x1, x2, y1, y2, certainty, spread)
        "bottom_right": np.array([-10, args.environment_size[1], args.environment_size[2], 10, 1, 1]),
        "top_right": np.array([-30, args.environment_size[1], 30, args.environment_size[3], 1, 1])}
    objs = []
    for key, value in static_obstacles.items():
        objs.append(StaticObstacle(args, coord=value, name="turnR " + key, timestep=0))
    return objs

def create_crossing4w(args):
    static_obstacles = {
        "bottom_left": np.array([args.environment_size[0], -10, args.environment_size[2], -10, 1, 1]), #(x1, x2, y1, y2, certainty, spread)
        "top_left": np.array([args.environment_size[0], -10, 10, args.environment_size[3], 1, 1]),
        "bottom_right": np.array([10, args.environment_size[1], args.environment_size[2], -10, 1, 1]),
        "top_right": np.array([10, args.environment_size[1], 10, args.environment_size[3], 1, 1])}
    objs = []
    for key, value in static_obstacles.items():
        objs.append(StaticObstacle(args, coord=value, name="crossing4w " + key, timestep=0))
    return objs

def create_obstacleBottom(args):
    obs = np.array([0, 3, -22, -17, 1, 20])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_obstacleSmallBM(args):
    obs = np.array([-1, 1, -20, -18, 1, 20])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_obstacleSmallL(args):
    obs = np.array([-7, -6, -15, -14, 1, 20])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_obstacleSmallR(args):
    obs = np.array([5, 6.5, -13, -12, 1, 20])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_obstacleSmallTM(args):
    obs = np.array([-1, 0, -5, -4, 1, 20])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_pedLR(args):
    ped = np.array([-7, -6, -5, -4, 0.8, 30])
    obj = DynamicObstacle(args, name="pedLR", coord=ped, velocity_x=1)
    return obj

def create_pedRL(args):
    ped = np.array([7, 8, -13, -12, 0.8, 30])
    obj = DynamicObstacle(args, name="pedLR", coord=ped, velocity_x=-1)
    return obj

def create_bikerBT(args):
    ped = np.array([-5, -4.9, -20, -18, 0.8, 30])
    obj = DynamicObstacle(args, name="bikerBT", coord=ped, velocity_y=2)
    return obj
