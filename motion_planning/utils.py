import numpy as np
import torch
from scipy.stats import multivariate_normal
from systems.utils import get_mesh_pos


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

def pos2gridpos(args, pos_x=None, pos_y=None, format='torch'):
    if pos_x is not None:
        if isinstance(pos_x, list):
            pos_x = np.array(pos_x)
        pos_x = np.array((pos_x - args.environment_size[0]) / (args.environment_size[1] - args.environment_size[0])) * \
                (args.grid_size[0]-1)
    if pos_y is not None:
        if isinstance(pos_y, list):
            pos_y = np.array(pos_y)
        pos_y = np.array((pos_y - args.environment_size[2]) / (args.environment_size[3] - args.environment_size[2])) * \
               (args.grid_size[1]-1)
    if format == 'torch':
        return torch.from_numpy(np.round(pos_x+0.001)).type(torch.LongTensor), torch.from_numpy(np.round(pos_y+0.001)).type(torch.LongTensor)
    else:
        return (np.round(pos_x + 0.001).astype(int)), (np.round(pos_y + 0.001).astype(int))


def gridpos2pos(args, pos_x=None, pos_y=None):
    if pos_x is not None:
        pos_x = (pos_x / (args.grid_size[0]-1)) * (args.environment_size[1] - args.environment_size[0]) + \
            args.environment_size[0]
    if pos_y is not None:
        pos_y = (pos_y / (args.grid_size[1]-1)) * (args.environment_size[3] - args.environment_size[2]) + \
            args.environment_size[2]
    return pos_x, pos_y


def shift_array(arr, step_x=0, step_y=0):
    result = torch.empty_like(arr)
    if step_x > 0:
        result[:step_x, :] = 0
        result[step_x:, :] = arr[:-step_x, :]
    elif step_x < 0:
        result[step_x:, :] = 0
        result[:step_x, :] = arr[-step_x:, :]
    arr = result
    if step_y > 0:
        result[:, step_y] = 0
        result[:, step_y:] = arr[:, -step_y]
    elif step_y < 0:
        result[:, step_y] = 0
        result[:, step_y] = arr[: -step_y:]
    else:
        result[:] = arr
    return result


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
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x_traj[0, 0, :], pos_y=x_traj[0, 1, :])
    grid = torch.zeros((args.grid_size[0], args.grid_size[1]))
    grid[gridpos_x.clamp(0, args.grid_size[0]-1), gridpos_y.clamp(0, args.grid_size[1]-1)] = 1
    return grid

def check_collision(grid_ego, grid_env, max_coll_sum):
    collision = False
    coll_grid = grid_ego * grid_env
    coll_sum = coll_grid.sum()  # coll_max = coll_grid.max()
    if coll_sum > max_coll_sum:
        collision = True  # if coll_max > args.coll_max:
    return collision, coll_sum  # return "coll_max", idx, coll_max

def pred2grid(x, rho, args):
    """average the density of points landing in the same bin and return normalized grid"""
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=x[:, 0, 0], pos_y=x[:, 1, 0])
    min_xbin = int(gridpos_x.min())
    min_ybin = int(gridpos_y.min())
    max_xbin = int(gridpos_x.max())
    max_ybin = int(gridpos_y.max())
    gridpos_x = gridpos_x.numpy()
    gridpos_y = gridpos_y.numpy()
    num_samples, _, _ = np.histogram2d(gridpos_x, gridpos_y, bins=[max_xbin - min_xbin + 1, max_ybin - min_ybin + 1],
                   range=[[min_xbin, max_xbin], [min_ybin, max_ybin]])
    density_sum, _, _ = np.histogram2d(gridpos_x, gridpos_y, bins=[max_xbin - min_xbin + 1, max_ybin - min_ybin + 1],
                   range=[[min_xbin, max_xbin], [min_ybin, max_ybin]], weights=rho[:, 0,0])
    density_mean = density_sum
    mask = num_samples > 0
    density_mean[mask] /= num_samples[mask]
    grid = torch.zeros((args.grid_size[0], args.grid_size[1], 1))
    grid[min_xbin:max_xbin+1, min_ybin:max_ybin+1, 0] = torch.from_numpy(density_mean) / density_mean.sum()
    return grid


def time2index(t, args):
    return int((t * args.factor_pred).round())

def get_mesh_sample_points(system, args):
    x_min = system.XE0_MIN.flatten()
    x_max = system.XE0_MAX.flatten()
    N = (x_max - x_min) / args.grid_wide + 1
    positions = get_mesh_pos(N, x_min=x_min, x_max=x_max)
    return N, positions

#
# def create_example_grid():
#     gridsize = (100, 100)
#     ntimesteps = 20
#     obstacles = (
#         (70, 75, 40, 45, 0.8, 20, "Ped1"), (0, 20, 0, 100, 1, 1, "static1"),
#         (80, 100, 0, 100, 1, 1, "static2"))  # (x1, x2, y1, y2, certainty, spread)
#     grid = Environment(gridsize=gridsize)
#
#     for coord in obstacles:
#         obj = OccupancyObject(gridsize=gridsize, name=coord[6],  pos=coord[0:4], certainty=coord[4], spread=coord[5], timestep=0)
#         grid.add_object(obj)
#
#     grid.objects[1].forward_occupancy(0,5)
#     grid.objects[1].plot_grid(5)
#     grid.plot_grid(5)
#
#     return grid
#
# def create_example_vehicle():
#     start = (60, 2)
#     goal = (70, 100)
#     T = 0.2
#     vehicle = Vehicle(start, goal, T)
#
#     return vehicle