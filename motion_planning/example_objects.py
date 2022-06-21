from motion_planning.simulation_objects import StaticObstacle, Environment, DynamicObstacle
import numpy as np


def create_environment(args, object_str_list=None, name="environment", timestep=0):
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
            if i >= num_static:
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
