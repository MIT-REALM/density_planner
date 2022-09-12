import numpy as np
import torch


def jacobian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Calculate vector-vector jacobian

    :param f: torch.Tensor
        bs x m x 1
    :param x: torch.Tensor
        bs x n x 1

    :return: J: torch.Tensor
        bs x m x n
    """
    #f = f + 0. * x.sum()  # to avoid the case that f is independent of x
    bs = x.shape[0]
    J = torch.zeros(bs, f.shape[1], x.shape[1]).type_as(x)
    for i in range(f.shape[1]):
        J[:, i, :] = torch.autograd.grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def approximate_derivative(function, x):
    """
    approximate derivative numerically
    """

    #numerical approximation
    bs = x.shape[0]
    num_state = x.shape[1]
    y = function(x)
    delta = 0.0001

    if y.shape[2] == 1:
        dydx = torch.zeros(bs, y.shape[1], num_state)
    else:
        dydx = torch.zeros(bs, y.shape[1], y.shape[2], num_state)

    for i in range(num_state):
        dx = torch.zeros_like(x)
        dx[:, i, 0] = delta
        if y.shape[2] == 1:
            dydx[:, :, [i]] = (function(x+dx)-y)/delta
        else:
            dydx[:, :, :, i] = (function(x + dx) - y) / delta
    return dydx

def load_controller(system_type):
    """
    load controller from specified path
    """

    # copied from C3M code "get_controller_wrapper" by Dawei Sun
    controller_path = 'data/trained_controller/controller_'+system_type+'.pth.tar'
    _controller = torch.load(controller_path, map_location=torch.device('cpu'))
    _controller.cpu()

    def controller(x, xe, uref):
        u = _controller(torch.from_numpy(x).float().view(1,-1,1), torch.from_numpy(xe).float().view(1,-1,1), torch.from_numpy(uref).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return u

    return controller


def get_mesh_pos(N, x_min=None, x_max=None):
    """
    Create a mesh over all state dimensions

    :param N: list or tensor
        dim_state: number of different mash values for each state dimension

    :return: x: torch.Tensor
        N x dim_state x 1: mesh over states
    """

    mesh_inputs = []
    if x_min is None:
        x_min = torch.zeros(len(N))
    if x_max is None:
        x_max = torch.ones(len(N))

    for i in range(len(N)):
        mesh_inputs.append(torch.linspace(x_min[i], x_max[i], int(N[i])))
    mesh_outputs = torch.meshgrid(*mesh_inputs, indexing='ij')
    for i, output in enumerate(mesh_outputs):
        if i == 0:
            positions = output.flatten().unsqueeze(-1)
        else:
            positions = torch.cat((positions, output.flatten().unsqueeze(-1)), 1)
    return positions


def listDict2dictList(list_dict):
    """
    convert list of dictionaries to a dictionary containing lists
    """
    dict_list = {key: [] for key in list_dict[0].keys()}
    for loss in list_dict:
        for key, val in loss.items():
            dict_list[key].append(val)
    return dict_list


def get_density_map(x, rho, args, log_density=False, type="LE", bins=None, bin_width=None, system=None):
    """
    compute the density distribution with the binning approach
    """
    if bins is None:
        bins = args.bin_number[:x.shape[1]]
    if bin_width is None:
        bin_width = (system.XE_MAX - system.XE_MIN)[0, :x.shape[1], 0] / bins

    if type != "MC" and log_density:
        rho -= rho.max()
        rho = torch.exp(rho)
    else:
        rho /= rho.max()

    range_bins = [[-bins[i] * bin_width[i] / 2, bins[i] * bin_width[i] / 2] for i in range(len(bins))]
    with torch.no_grad():
        density_mean, _ = np.histogramdd(x.numpy(), bins=bins, weights=rho.numpy(), range=range_bins)
    mask = density_mean > 0
    if type != "MC":
        with torch.no_grad():
            num_samples, _ = np.histogramdd(x.numpy(), bins=bins, range=range_bins)
        density_mean[mask] /= num_samples[mask]
    density_mean[mask] = density_mean[mask] / density_mean[mask].sum()
    return density_mean, range_bins


def sample_binpos(sample_size, bin_numbers, bin_width=None):
    """
    sample grid cell and compute the corresponding position
    """

    binpos = torch.zeros(sample_size, len(bin_numbers))
    xepos = None
    s = int(sample_size/2)
    for i, b in enumerate(bin_numbers):
        binpos[:s, i] = torch.randint(0, b, (s,))
        binpos[s:, i] = torch.randint(int(b/4), int(np.ceil(3*b/4)), (s,))  # get more samples in the middle
    if bin_width is not None:
        xepos = binpos * bin_width - 0.5 * bin_width * (torch.tensor(bin_numbers) - 1)
    return binpos.long(), xepos
