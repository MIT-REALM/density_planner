import numpy as np
import torch
from scipy.stats import multivariate_normal
from systems.utils import get_mesh_pos
from datetime import datetime
import os
import logging
import sys
import shutil


class MultivariateGaussians():
    """
    class for the pdf
    """
    def __init__(self, means, cov_diags, weights, x_min, x_max):
        """
        initialize parameters

        :param means:       mean vectors of the Gaussians
        :param cov_diags:   diagonal values of the covariance matrices
        :param weights:     weights for the Gaussians
        :param x_min:       lower bound for the states
        :param x_max:       upper bound
        """
        self.means = means
        self.cov_diags = cov_diags
        self.weights = weights
        self.min = x_min
        self.max = x_max

    def __call__(self, x):
        """
        compute the probability for sample x

        :param x: sample state x
        :return: probability of x
        """
        prob = torch.zeros(x.shape[0])
        for i, w in enumerate(self.weights):
            prob += w * multivariate_normal.pdf(x[:,:, 0], self.means[i, :], torch.diag(self.cov_diags[i, :]))

        mask = torch.logical_or(torch.any(x[:, :, 0] < self.min[[0], :, 0], 1),
                                torch.any(x[:, :, 0] > self.max[[0], :, 0], 1))
        prob[mask] = 0
        return prob


def initialize_logging(args, name):
    """
    create folder for saving plots and create logger
    """
    path_log = make_path(args.path_plot_motion, name)
    shutil.copyfile('hyperparams.py', path_log + 'hyperparams.py')
    shutil.copyfile('motion_planning/simulation_objects.py', path_log + 'simulation_objects.py')
    shutil.copyfile('motion_planning/MotionPlanner.py', path_log + 'MotionPlanner.py')
    shutil.copyfile('motion_planning/MotionPlannerGrad.py', path_log + 'MotionPlannerGrad.py')
    shutil.copyfile('motion_planning/MotionPlannerNLP.py', path_log + 'MotionPlannerNLP.py')
    shutil.copyfile('motion_planning/plan_motion.py', path_log + 'plan_motion.py')
    if args.mp_use_realEnv:
        shutil.copyfile('env/environment.py', path_log + 'environment.py')

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[
                            logging.FileHandler(path_log + '/logfile.txt'),
                            logging.StreamHandler(sys.stdout)])
    return path_log


def pos2gridpos(args, pos_x=None, pos_y=None):
    """
    transform position in real-world coordinates to position in grid coordinates

    :param args:    settings
    :param pos_x:   x coordinates which should be converted
    :param pos_y:   y coordinates which should be converted
    :return: x and y position in grid coordinates
    """
    if pos_x is not None:
        if isinstance(pos_x, list):
            pos_x = torch.from_numpy(np.array(pos_x))
        pos_x = ((pos_x - args.environment_size[0]) / (args.environment_size[1] - args.environment_size[0])) * \
                (args.grid_size[0]-1)
        if torch.is_tensor(pos_x):
            pos_x = (torch.round(pos_x + 0.001)).long()
        else:
            pos_x = (np.round(pos_x + 0.001).astype(int))
    if pos_y is not None:
        if isinstance(pos_y, list):
            pos_y = torch.from_numpy(np.array(pos_y))
        pos_y = ((pos_y - args.environment_size[2]) / (args.environment_size[3] - args.environment_size[2])) * \
               (args.grid_size[1]-1)
        if torch.is_tensor(pos_y):
            pos_y = (torch.round(pos_y + 0.001)).long()
        else:
            pos_y = (np.round(pos_y + 0.001).astype(int))
    return pos_x, pos_y


def gridpos2pos(args, pos_x=None, pos_y=None):
    """
    transform position in grid coordinates to position in real-world coordinates

    :param args:    settings
    :param pos_x:   x coordinates which should be converted
    :param pos_y:   y coordinates which should be converted
    :return: x and y position in real-world coordinates
    """

    if pos_x is not None:
        pos_x = (pos_x / (args.grid_size[0]-1)) * (args.environment_size[1] - args.environment_size[0]) + \
            args.environment_size[0]
    if pos_y is not None:
        pos_y = (pos_y / (args.grid_size[1]-1)) * (args.environment_size[3] - args.environment_size[2]) + \
            args.environment_size[2]
    return pos_x, pos_y


def bounds2array(env, args):
    """
    get bounds of environment obstacles in real-world coordinates

    :param env:  environment
    :param args: settings
    :return: arrays with the bounds
    """
    bounds_array = np.zeros((len(env.objects), 4, env.grid.shape[2]))
    for t in range(env.grid.shape[2]):
        for k in range(len(env.objects)):
            bounds_x, bounds_y = gridpos2pos(args, pos_x=env.objects[k].bounds[t][:2], pos_y=env.objects[k].bounds[t][2:])
            bounds_array[k, :2, t] = bounds_x
            bounds_array[k, 2:, t] = bounds_y
    return bounds_array


def shift_array(grid, step_x=0, step_y=0, fill=0):
    """
    shift array or tensor

    :param grid:    array/ tensor which should be shifted
    :param step_x:  shift in x direction
    :param step_y:  shift in y direction
    :param fill:    value which should be used to fill the new cells
    :return: shifted array / tensor
    """

    result = torch.zeros_like(grid)

    if step_x > 0:
        result[:step_x, :] = fill
        result[step_x:, :] = grid[:-step_x, :]
    elif step_x < 0:
        result[step_x:, :] = fill
        result[:step_x, :] = grid[-step_x:, :]
    else:
        result = grid

    result_new = torch.zeros_like(grid)
    if step_y > 0:
        result_new[:, :step_y] = fill
        result_new[:, step_y:] = result[:, :-step_y]
    elif step_y < 0:
        result_new[:, step_y:] = fill
        result_new[:, :step_y] = result[:, -step_y:]
    else:
        result_new = result
    return result_new


def enlarge_grid(grid, width):
    """
    compute the occupation map with padded obstacles
    :param grid:  occupation map
    :param width: size of the padding
    :return: padded occupation map
    """
    grid_enlarged = grid
    for i in range(width):
        grid_enlarged += shift_array(grid, step_x=1)
        grid_enlarged += shift_array(grid, step_x=-1)
    grid = grid_enlarged
    for i in range(width):
        grid_enlarged += shift_array(grid, step_y=1)
        grid_enlarged += shift_array(grid, step_y=-1)
    return torch.clamp(grid_enlarged, 0, 1)


def compute_gradient(grid, step=1):
    """
    compute the gradient in x and y direction of a given tensor
    :param grid: tensor
    :param step: step size for the gradient computation
    :return: gradient in x direction, gradient y direction
    """

    grid_gradientX = (shift_array(grid, step_x=step, fill=1) - shift_array(grid, step_x=-step, fill=1)) / (2 * step)
    grid_gradientY = (shift_array(grid, step_y=step, fill=1) - shift_array(grid, step_y=-step, fill=1)) / (2 * step)
    return grid_gradientX, grid_gradientY


def sample_pdf(system, num, spread=0.3):
    """
    create random pdf

    :param system:  system
    :param num:     number of gaussians
    :param spread:  spread of the mean vecors of the Gaussians
    :return: pdf
    """
    weights = torch.rand(num)
    means = spread * torch.randn(num, system.DIM_X)
    cov_diags = 5 * torch.rand(num, system.DIM_X)
    pdf = MultivariateGaussians(means, cov_diags, weights / weights.sum(), system.XE_MIN, system.XE_MAX)
    return pdf


def traj2grid(x_traj, args):
    """
    convert trajectory to an occupation grid

    :param x_traj:  state trajectory
    :param args:    settings
    :return: occupation grid
    """
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x_traj[:, 0, :], pos_y=x_traj[:, 1, :])
    grid = torch.zeros((args.grid_size[0], args.grid_size[1]))
    grid[gridpos_x.clamp(0, args.grid_size[0]-1), gridpos_y.clamp(0, args.grid_size[1]-1)] = 1
    return grid


def check_collision(grid_ego, grid_env, max_coll_sum, return_coll_pos=False):
    """
    test if ego vehicle collides with obstacles

    :param grid_ego:        occupation map of the ego vehicle
    :param grid_env:        occupation map of the environment
    :param max_coll_sum:    threshold for the admissible collision probability
    :param return_coll_pos: True if also the position of the collision should be computed
    :return:
    """
    collision = False
    coll_grid = grid_ego * grid_env
    coll_sum = coll_grid.sum()

    coll_pos_prob = None
    if coll_sum > max_coll_sum:
        collision = True  # if coll
        if return_coll_pos:
            coll_pos = coll_grid.nonzero(as_tuple=True)
            coll_prob = coll_grid[coll_pos[:][0], coll_pos[:][1]]
            coll_pos_prob = torch.stack((coll_pos[:][0], coll_pos[:][1], coll_prob[:]))
    return collision, coll_sum, coll_pos_prob


def pred2grid(x, rho, args, return_gridpos=False):
    """
    average the density of points landing in the same bin and return normalized grid
    """

    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x[:, 0, 0], pos_y=x[:, 1, 0])
    gridpos_x = torch.clamp(gridpos_x, 0, args.grid_size[0])
    gridpos_y = torch.clamp(gridpos_y, 0, args.grid_size[1])
    min_xbin = int(gridpos_x.min())
    min_ybin = int(gridpos_y.min())
    max_xbin = int(gridpos_x.max())
    max_ybin = int(gridpos_y.max())

    gridpos = torch.cat((gridpos_x.unsqueeze(-1), gridpos_y.unsqueeze(-1)), dim=1)
    num_samples, _ = torch.histogramdd(gridpos.type(torch.FloatTensor), bins=[max_xbin - min_xbin + 1, max_ybin - min_ybin + 1],
                   range=[min_xbin, max_xbin, min_ybin, max_ybin])
    density_sum, _ = torch.histogramdd(gridpos.type(torch.FloatTensor), bins=[max_xbin - min_xbin + 1, max_ybin - min_ybin + 1],
                   weight=rho[:, 0,0], range=[min_xbin, max_xbin, min_ybin, max_ybin])
    density_mean = density_sum
    mask = num_samples > 0
    density_mean[mask] /= num_samples[mask]
    grid = torch.zeros((args.grid_size[0], args.grid_size[1], 1))
    grid[min_xbin:max_xbin+1, min_ybin:max_ybin+1, 0] = density_mean / density_mean.sum()
    if return_gridpos:
        return grid, gridpos
    return grid


def get_closest_free_cell(gridpos, grid_env, args):
    """
    compute the closest cell where occupation probability is zero

    :param gridpos: current grid position
    :param grid_env:occupation map of the environment
    :param args:    settings
    :return: grid position of the closest free cell
    """

    gridpos = gridpos.long()
    for i in range(1, min(args.grid_size)):
        min_x = max(gridpos[0] - i, 0)
        max_x = min(gridpos[0] + i + 1, args.grid_size[0])
        min_y = max(gridpos[1] - i, 0)
        max_y = min(gridpos[1] + i + 1, args.grid_size[1])

        if torch.any(grid_env[min_x:max_x, gridpos[1]] == 0):
            free_cell = (grid_env[min_x:max_x, gridpos[1]] == 0).nonzero(as_tuple=True)
            return torch.tensor([min_x+free_cell[0][0], gridpos[1]])

        if torch.any(grid_env[min_x:max_x, min_y:max_y] == 0):
            free_cell = (grid_env[min_x:max_x, min_y:max_y] == 0).nonzero(as_tuple=True)
            return torch.tensor([min_x+free_cell[0][0], min_y+free_cell[1][0]])
    return None


def get_closest_obs_cell(gridpos, grid_env, args):
    """
    compute the closest cell where occupation probability is nonzero

    :param gridpos: current grid position
    :param grid_env:occupation map of the environment
    :param args:    settings
    :return: grid position of the closest obstacle cell
    """

    gridpos = gridpos.long()
    for i in range(1, args.max_obsDist):
        min_x = max(gridpos[0] - i, 0)
        max_x = min(gridpos[0] + i + 1, args.grid_size[0])
        min_y = max(gridpos[1] - i, 0)
        max_y = min(gridpos[1] + i + 1, args.grid_size[1])

        if torch.any(grid_env[min_x:max_x, gridpos[1]] != 0):
            obs_cell = (grid_env[min_x:max_x, gridpos[1]] != 0).nonzero(as_tuple=True)
            return torch.tensor([min_x+obs_cell[0][0], gridpos[1]])

        if torch.any(grid_env[min_x:max_x, min_y:max_y] != 0):
            obs_cell = (grid_env[min_x:max_x, min_y:max_y] != 0).nonzero(as_tuple=True)
            return torch.tensor([min_x+obs_cell[0][0], min_y+obs_cell[1][0]])
    return None


def time2idx(t, args, short=True):
    """
    convert continuous time point to the number of time steps

    :param t:       point in time
    :param args:    settings
    :param short:   True if we consider short time step
    :return: number of time steps
    """
    if isinstance(t, list):
        t = torch.from_numpy(np.array(t))
    if short:
        return int((t / (args.dt_sim * args.factor_pred)).round())
    else:
        return int((t / args.dt_sim).round())


def idx2time(idx, args, short=True):
    """
    convert number of time steps to continuous time

    :param t:       point in time
    :param args:    settings
    :param short:   True if we consider short time step
    :return: point in time
    """

    if isinstance(idx, list):
        idx = torch.from_numpy(np.array(idx))
    if short:
        return idx * args.dt_sim * args.factor_pred
    else:
        return idx * args.dt_sim


def get_mesh_sample_points(system, args):
    """
    sample initial states in a regular mesh

    :param system:  system
    :param args:    settings
    :return: dimension of the mesh and sampled states
    """
    x_min = system.XE0_MIN.flatten()
    x_max = system.XE0_MAX.flatten()
    N = (x_max - x_min) / args.grid_wide + 1
    positions = get_mesh_pos(N, x_min=x_min, x_max=x_max)
    return N, positions


def make_path(path0, name):
    """
    create folder

    :param path0:   path to the folder
    :param name:    folder name
    :return:
    """
    path = path0 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + name + "/"
    os.makedirs(path)
    return path


def get_cost_increase(methods, results, thr_coll=10, thr_goal=20, max_iter=None):
    """
    compute and print the evaluation criteria collision risk increase, goal cost increase, input cost increase and
    failure rate

    :param methods:     list with all methods which should be evaluated
    :param results:     motion planning data
    :param thr_coll:    threshold for accepted collision cost
    :param thr_goal:    threshold for accepted goal cost
    :param max_iter:    last environment index which should be considered for the computations
    :return: motion planning data extended with the evaluation criteria
    """
    with torch.no_grad():
        # look for failures (cost_coll > failure_thr)
        for method in methods:
            if max_iter is None:
                num_iter = len(results[method]["cost"])
            else:
                num_iter = max_iter
            for k in range(num_iter):
                if results[method]["cost"][k] is not None and (results[method]["cost"][k]["cost_coll"] > thr_coll or
                                                               results[method]["cost"][k]["cost_goal"] > thr_goal or
                                                               results[method]["cost"][k]["cost_bounds"] > 1e-4):
                    results[method]["cost"][k] = None
                    results[method]["num_valid"] -= 1

        # set initial cost_increase to zero
        for method in methods:
            for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref"]:
                results[method][s] = 0

        # compute cost_increase
        for k in range(num_iter):
            for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref"]:
                cost_min = np.inf
                for method in methods:
                    if results[method]["cost"][k] is not None and \
                            results[method]["cost"][k][s] < cost_min:
                        cost_min = results[method]["cost"][k][s].item()
                for method in methods:
                    if results[method]["cost"][k] is not None:
                        results[method][s] += results[method]["cost"][k][s] - cost_min
    print_cost_increase(methods, results)
    return results


def print_cost_increase(methods, results):
    """
    print the evaluation criteria

    :param methods:     list with all methods which should be evaluated
    :param results:     motion planning data
    """
    for method in methods:
        print("%s: number valid: %d" % (method, results[method]["num_valid"]))
        for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "sum_time"]:
            print("%s: %s: %.2f" % (method, s, results[method][s] / results[method]["num_valid"]))


def get_cost_table(methods, results, max_iter=None):
    """
    print the motion planning results in latex-format


    :param methods:     list with all methods which should be evaluated
    :param results:     motion planning data
    :param max_iter:    last environment index which should be printed
    """
    print("#### TABLE:")
    for s in ["cost_coll", "cost_goal", "cost_bounds", "cost_uref", "time"]:
        print("#### %s:" % s)
        if max_iter is None:
            num_iter = len(results[methods[0]]["cost"])
        else:
            num_iter = max_iter
        for l in range(num_iter):
            print("%d" % l, end=" & ")
            for method in methods:
                if results[method]["cost"][l] is not None:
                    if s == "time":
                        if "grad" in method:
                            continue
                        print("%.2f" % results[method]["time"][l], end=" & ")
                    else:
                        print("%.3f" % results[method]["cost"][l][s].item(), end=" & ")
                else:
                    print(" - ", end=" & ")
            print(" \\ ")


def find_start_goal(env, args):
    """
    find random start and goal state in the free space of the environment

    :param env:  environment
    :param args: settings
    :return: start state, goal state
    """

    iter = 0
    valid = False

    while not valid:
        if env.config["dataset"]["recording"] >= 18 and env.config["dataset"]["recording"] <= 29:
            if np.random.randint(0, 2) == 0:
                pos_0 = np.array([-20, -10]) + np.array([20, 7]) * np.random.rand(2)
            else:
                pos_0 = np.array([-15, -30]) + np.array([15, 17]) * np.random.rand(2)
            if np.random.randint(0, 2) == 0:
                pos_N = np.array([10, -20]) + np.array([30, 8]) * np.random.rand(2)
            else:
                pos_N = np.array([0, -5]) + np.array([20, 7]) * np.random.rand(2)
        elif env.config["dataset"]["recording"] >= 30:
            pos_0 = np.array([-15, -15]) + 10 * np.random.rand(2)
            pos_N = np.array([25, 15]) + 10 * np.random.rand(2)  # env.generate_random_waypoint(0)
        elif env.config["dataset"]["recording"] >= 7 and env.config["dataset"]["recording"] <= 10:
            pos_0 = np.array([0, -40]) + np.array([10, 20]) * np.random.rand(2)
            pos_N = np.array([30, -10]) + 10 * np.random.rand(2) 
        theta_0 = 1.6 * np.random.rand(1)
        v_0 = 1 + 8 * np.random.rand(1)
        valid = check_start_goal(env.grid, pos_0, pos_N, theta_0, v_0, args)
        iter += 1
        if iter > 2000 and env.config["dataset"]["recording"] != 26: # to make thesis results reproducible
            return None, None

    return torch.tensor([pos_0[0], pos_0[1], theta_0[0], v_0[0], 0]).reshape(1, -1, 1).type(torch.FloatTensor), torch.tensor([pos_N[0], pos_N[1], 0, 0, 0]).reshape(1, -1, 1)


def check_start_goal(grid, pos_0, pos_N, theta_0, v_0, args):
    """
    check if suggested start and goal state is valid

    :param grid:    environment occupation map
    :param pos_0:   position of the start state
    :param pos_N:   position of the goal state
    :param theta_0: initial heading
    :param v_0:     initial velocity
    :param args:    settings
    :return: True if start and goal is valid
    """
    dist_start_goal = np.sqrt((pos_0[0] - pos_N[0]) ** 2 + (pos_0[1] - pos_N[1]) ** 2)
    pos_1s = pos_0 + np.array([v_0[0] * np.cos(theta_0[0]), v_0[0] * np.sin(theta_0[0])])
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[pos_0[0], pos_0[0] - 0.5, pos_0[0] + 0.5, pos_1s[0], pos_N[0],
                                                    pos_N[0] - 0.5, pos_N[0] + 0.5],
                                       pos_y=[pos_0[1], pos_0[1] - 0.5, pos_0[1] + 0.5, pos_1s[1], pos_N[1],
                                              pos_N[1] - 0.5, pos_N[1] + 0.5])
    for k in range(len(gridpos_x)):
        if gridpos_x[k] < 0 or gridpos_y[k] < 0 or gridpos_x[k] > grid.shape[0] - 1 or gridpos_y[k] > grid.shape[1] - 1:
            return False
    density_0 = grid[gridpos_x[0], gridpos_y[0], 0]
    density_01 = grid[gridpos_x[1], gridpos_y[1], 0]
    density_02 = grid[gridpos_x[2], gridpos_y[1], 0]
    density_03 = grid[gridpos_x[1], gridpos_y[2], 0]
    density_04 = grid[gridpos_x[2], gridpos_y[2], 0]
    density_0 = max(density_0, density_01, density_02, density_03, density_04)
    density_1s = grid[gridpos_x[3], gridpos_y[3], 10]
    density_N = grid[gridpos_x[4], gridpos_y[4], 100]
    density_N1 = grid[gridpos_x[5], gridpos_y[5], 100]
    density_N2 = grid[gridpos_x[6], gridpos_y[5], 100]
    density_N3 = grid[gridpos_x[5], gridpos_y[6], 100]
    density_N4 = grid[gridpos_x[6], gridpos_y[6], 100]
    density_N = max(density_N, density_N1, density_N2, density_N3, density_N4)
    return not (dist_start_goal < 10 or dist_start_goal > 70 or density_0 > 0.1 or density_1s > 0.3 or density_N > 0.1)


def convert_color(s):
    """
    convert color to rgba

    :param s: old color format (string with hexadecimal code or rgb values)
    :return: rgba values
    """

    if type(s) == str:
        return np.array([[float(int(s[1:3], 16)) / 255, float(int(s[3:5], 16)) / 255, float(int(s[5:], 16)) / 255, 1]])
    return np.array([[s[0], s[1], s[2], 1]])

