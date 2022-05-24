import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal
from systems.utils import get_mesh_pos
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap


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
            prob += w * multivariate_normal.pdf(x[:,:], self.means[i, :], torch.diag(self.cov_diags[i, :]))

        mask = torch.logical_or(torch.any(x[:, :] <= self.min[[0], :, 0], 1),
                                torch.any(x[:, :] >= self.max[[0], :, 0], 1))
        prob[mask] = 0
        return prob

class DensityPredictor():
    def __init__ (self, xref0, u_params, pdf0, system, args):
        self.xref0 = xref0
        self.u_params = u_params
        self.xref_traj = system.u_params2xref_traj(xref0, u_params, args)
        self.pdf0 = pdf0
        self.sampling = args.sampling
        self.sample_size = args.sample_size
        self.system = system
        self.model = self.load_predictor(args, system.DIM_X)

    def __call__(self, t_vec, system, args):
        # 1. sample batch xe0
        if self.sampling == 'random':
            xe0 = torch.rand(self.sample_size, system.DIM_X, 1) * (system.XE0_MAX - system.XE0_MIN) + system.XE0_MIN
        else:
            _, xe0 = get_mesh_sample_points(system, args)
            xe0 = xe0.unsqueeze(-1)
        rho0 = self.pdf0(xe0)

        # 2. predict xe(t) and rho(t) for times t
        xe_nn = torch.zeros(xe0.shape[0], system.DIM_X, len(t_vec))
        rho_nn = torch.zeros_like(xe0.shape[0], 1, len(t_vec))
        for t in t_vec:
            i = t * args.factor_pred
            xe_nn[:, :, [i]], rho_nn[:, :, [i]] = get_nn_prediction(self.model, xe0, self.xref0, t, self.u_params, args)
        rho_nn *= rho0 #.unsqueeze(-1).unsqueeze(-1)

        # 3. compute x_nn
        x_nn = xe_nn + self.xref_traj


        # 4. interpolate x(t), rho(t) to get pdf at t


        # 5. return grid with final occupation probabilities

    def load_predictor(args, dim_x):
        _, num_inputs = load_inputmap(dim_x, args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True)
        model.eval()
        return model

def plot_grid(object, args, timestep=None):
    if timestep is None:
        timestep = object.current_timestep
    x_wide = max((args.environment_size[1] - args.environment_size[0]) / 10, 3)
    y_wide = (args.environment_size[3] - args.environment_size[2]) / 10
    plt.figure(figsize=(x_wide, y_wide))
    plt.pcolormesh(object.grid[:, :, timestep].T, cmap='binary')
    plt.axis('scaled')

    ticks_x = np.concatenate((np.arange(0, args.environment_size[1]+1, 10), np.arange(-10, args.environment_size[0]-1, -10)), 0)
    ticks_y = np.concatenate((np.arange(0, args.environment_size[3]+1, 10), np.arange(-10, args.environment_size[2]-1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)

    plt.title(f"{object.name} at timestep={timestep}")
    plt.tight_layout()
    plt.show()

def pos2gridpos(args, pos_x=None, pos_y=None, format='torch'):
    if pos_x is not None:
        if isinstance(pos_x, list):
            pos_x = np.array(pos_x)
        pos_x = np.array((pos_x - args.environment_size[0]) / (args.environment_size[1] - args.environment_size[0])) * \
               args.grid_size[0]
    if pos_y is not None:
        if isinstance(pos_y, list):
            pos_y = np.array(pos_y)
        pos_y = np.array((pos_y - args.environment_size[2]) / (args.environment_size[3] - args.environment_size[2])) * \
               args.grid_size[1]
    if format == 'torch':
        return torch.from_numpy(np.round(pos_x+0.001).astype(int)), torch.from_numpy(np.round(pos_y+0.001).astype(int))
    else:
        return (np.round(pos_x + 0.001).astype(int)), (np.round(pos_y + 0.001).astype(int))


def gridpos2pos(args, pos_x=None, pos_y=None):
    if pos_x is not None:
        pos_x = (pos_x / args.grid_size[0]) * (args.environment_size[1] - args.environment_size[0]) + \
            args.environment_size[0]
    if pos_y is not None:
        pos_y = (pos_y / args.grid_size[1]) * (args.environment_size[3] - args.environment_size[2]) + \
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


def get_pdf_grid(grid, grid_pos_x, grid_pos_y, certainty, spread, pdf_form='square'):
    normalise = False
    if certainty is None:
        certainty = 1
        normalise = True
    if pdf_form == 'square':
        for i in range(int(spread)):
            grid[grid_pos_x[0] - i:grid_pos_x[1] + i, grid_pos_y[0] - i:grid_pos_y[1] + i] \
                += certainty / spread
                #= grid[grid_pos_x[0] - i:grid_pos_x[1] + i, grid_pos_y[0] - i:grid_pos_y[1] + i, :] + certainty / spread
    else:
        raise(NotImplementedError)

    if normalise:
        grid = grid / grid.sum()
    return grid


def sample_pdf(system, num):
    weights = torch.rand(num)
    means = 0.3 * torch.randn(num, system.DIM_X)
    cov_diags = torch.rand(num, system.DIM_X)
    pdf = MultivariateGaussians(means, cov_diags, weights / weights.sum(), system.XE_MIN, system.XE_MAX)
    return pdf


def pdf2grid(pdf, xref0, system, args):
    N, positions = get_mesh_sample_points(system, args)
    probs = pdf(positions)
    grid_pos_x, grid_pos_y = pos2gridpos(args, positions[:, 0]+xref0[0], positions[:, 1]+xref0[1], format='np')
    grid = torch.zeros((args.grid_size[0], args.grid_size[1], 1))
    step = int(N[2] * N[3])
    for i in range(int(N[0] * N[1])):
        grid[(grid_pos_x[i * step]), (grid_pos_y[i * step])] += probs[i*step:(i+1)*step].sum()
    grid /= grid.sum() #N[3] * N[4]
    return grid


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