from motion_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np


def create_environment(object_str_list, args, name="environment", timestep=0):
    objects = []
    for obj_str in object_str_list:
        obj = globals()["create_" + obj_str](args)
        if isinstance(obj, list):
            objects += obj
        else:
            objects.append(obj)
    environment = Environment(objects, args, name=name)
    if timestep > 0:
        environment.forward_occupancy(step_size=timestep)
    return environment

def create_street(args):
    static_obstacles = {
        "left": np.array([args.environment_size[0], -10, args.environment_size[2], args.environment_size[3], 1, 1]), #(x1, x2, y1, y2, certainty, spread)
        "right": np.array([10, args.environment_size[1], args.environment_size[2], args.environment_size[3], 1, 1])}
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
    obs = np.array([0, 5, -20, -15, 1, 1])
    obj = StaticObstacle(args, name="obstacleBottom", coord=obs)
    return obj

def create_pedLR(args):
    ped = np.array([-11, -9, -11, -9, 0.8, 20])
    obj = DynamicObstacle(args, name="pedLR", coord=ped, velocity_x=1)
    return obj

def create_pedRL(args):
    ped = np.array([9, 11, -21, -19, 0.8, 20])
    obj = DynamicObstacle(args, name="pedLR", coord=ped, velocity_x=-1)
    return obj
