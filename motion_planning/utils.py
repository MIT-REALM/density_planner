import numpy as np
import torch
from scipy.stats import multivariate_normal
from systems.utils import get_mesh_pos
from datetime import datetime
import os

class MultivariateGaussians():
    def __init__(self, means, cov_diags, weights, x_min, x_max):
        self.means = means
        self.cov_diags = cov_diags
        self.weights = weights
        self.min = x_min
        self.max = x_max
        ##to-do: normalize pdfs (values outside of limits are zero)

    def __call__(self, x):
        prob = torch.zeros(x.shape[0])
        for i, w in enumerate(self.weights):
            prob += w * multivariate_normal.pdf(x[:,:, 0], self.means[i, :], torch.diag(self.cov_diags[i, :]))

        mask = torch.logical_or(torch.any(x[:, :, 0] < self.min[[0], :, 0], 1),
                                torch.any(x[:, :, 0] > self.max[[0], :, 0], 1))
        prob[mask] = 0
        return prob

def pos2gridpos(args, pos_x=None, pos_y=None):
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
    if pos_x is not None:
        pos_x = (pos_x / (args.grid_size[0]-1)) * (args.environment_size[1] - args.environment_size[0]) + \
            args.environment_size[0]
    if pos_y is not None:
        pos_y = (pos_y / (args.grid_size[1]-1)) * (args.environment_size[3] - args.environment_size[2]) + \
            args.environment_size[2]
    return pos_x, pos_y


def bounds2array(env, args):
    bounds_array = np.zeros((len(env.objects), 4, env.grid.shape[2]))
    for t in range(env.grid.shape[2]):
        for k in range(len(env.objects)):
            bounds_x, bounds_y = gridpos2pos(args, pos_x=env.objects[k].bounds[t][:2], pos_y=env.objects[k].bounds[t][2:])
            bounds_array[k, :2, t] = bounds_x
            bounds_array[k, 2:, t] = bounds_y
    return bounds_array


def shift_array(grid, step_x=0, step_y=0, fill=0):
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


def enlarge_grid(grid, wide):
    grid_enlarged = grid #.clone().detach()
    for i in range(wide):
        grid_enlarged += shift_array(grid, step_x=1)
        grid_enlarged += shift_array(grid, step_x=-1)
    grid = grid_enlarged #.clone().detach()
    for i in range(wide):
        grid_enlarged += shift_array(grid, step_y=1)
        grid_enlarged += shift_array(grid, step_y=-1)
    return torch.clamp(grid_enlarged, 0, 1)

def compute_gradient(grid, step=1):
    grid_gradientX = (shift_array(grid, step_x=step, fill=1) - shift_array(grid, step_x=-step, fill=1)) / (2 * step)
    grid_gradientY = (shift_array(grid, step_y=step, fill=1) - shift_array(grid, step_y=-step, fill=1)) / (2 * step)
    return grid_gradientX, grid_gradientY

def sample_pdf(system, num, spread=0.3):
    weights = torch.rand(num)
    means = spread * torch.randn(num, system.DIM_X)
    cov_diags = 5 * torch.rand(num, system.DIM_X)
    pdf = MultivariateGaussians(means, cov_diags, weights / weights.sum(), system.XE_MIN, system.XE_MAX)
    return pdf


# def pdf2grid(pdf, xref0, system, args):
#     N, positions = get_mesh_sample_points(system, args)
#     probs = pdf(positions)
#     grid_pos_x, grid_pos_y = pos2gridpos(args, positions[:, 0]+xref0[0, 0, 0], positions[:, 1]+xref0[0, 1, 0], format='np')
#     grid = torch.zeros((args.grid_size[0], args.grid_size[1], 1))
#     step = int(N[2] * N[3])
#     for i in range(int(N[0] * N[1])):
#         grid[(grid_pos_x[i * step]), (grid_pos_y[i * step])] += probs[i*step:(i+1)*step].sum()
#     grid /= grid.sum() #N[3] * N[4]
#     return grid
#--> pred2grid is 3times faster

def traj2grid(x_traj, args):
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x_traj[:, 0, :], pos_y=x_traj[:, 1, :])
    grid = torch.zeros((args.grid_size[0], args.grid_size[1]))
    grid[gridpos_x.clamp(0, args.grid_size[0]-1), gridpos_y.clamp(0, args.grid_size[1]-1)] = 1
    return grid

def check_collision(grid_ego, grid_env, max_coll_sum, return_coll_pos=False):
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
    """average the density of points landing in the same bin and return normalized grid"""
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
    if isinstance(t, list):
        t = torch.from_numpy(np.array(t))
    if short:
        return int((t / (args.dt_sim * args.factor_pred)).round())
    else:
        return int((t / args.dt_sim).round())

def idx2time(idx, args, short=True):
    if isinstance(idx, list):
        idx = torch.from_numpy(np.array(idx))
    if short:
        return idx * args.dt_sim * args.factor_pred
    else:
        return idx * args.dt_sim

def get_mesh_sample_points(system, args):
    x_min = system.XE0_MIN.flatten()
    x_max = system.XE0_MAX.flatten()
    N = (x_max - x_min) / args.grid_wide + 1
    positions = get_mesh_pos(N, x_min=x_min, x_max=x_max)
    return N, positions

def make_path(path0, name):
    path = path0 + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_" + name + "/"
    os.makedirs(path)
    return path