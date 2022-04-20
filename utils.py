import torch
import os
import yaml


def jacobian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Calculate vector-vector jacobian

    Parameters
    ----------
    f: torch.Tensor
        bs x m x 1
    x: torch.Tensor
        bs x n x 1

    Returns
    -------
    J: torch.Tensor
        bs x m x n
    """
    #f = f + 0. * x.sum()  # to avoid the case that f is independent of x
    bs = x.shape[0]
    J = torch.zeros(bs, f.shape[1], x.shape[1]).type_as(x)
    for i in range(f.shape[1]):
        J[:, i, :] = torch.autograd.grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J

def approximate_derivative(function, x):
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

def load_controller(task):
    # copied from C3M code "get_controller_wrapper" by Dawei Sun
    controller_path = 'data/trained_controller/controller_'+task+'.pth.tar'
    _controller = torch.load(controller_path, map_location=torch.device('cpu'))
    _controller.cpu()

    def controller(x, xe, uref):
        u = _controller(torch.from_numpy(x).float().view(1,-1,1), torch.from_numpy(xe).float().view(1,-1,1), torch.from_numpy(uref).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return u

    return controller

def load_data(task):
    data_path = 'density_data/data_' + task

    return data


def get_grid(N):
    """
    Create a grid over all state dimensions

    Parameters
    ----------
    N: list or tensor
        dim_state: number of different grid values for each state dimension

    Returns
    -------
    x: torch.Tensor
        N x dim_state x 1: grid over states
    """

    dim_x = len(N)
    x = torch.arange(0, 1+1/(N[0]-1)-1e-10, 1/(N[0]-1)).unsqueeze(-1)
    for i in range(1, dim_x):
        len_x = x.shape[0]
        for j in torch.arange(0, 1+1/(N[i]-1)-1e-10, 1/(N[i]-1)):
            if j == 0:
                y = j * torch.ones(len_x, 1)
            else:
                y = torch.cat((y, j*torch.ones(len_x, 1)),0)
        x = torch.cat((x.repeat(N[i],1), y), 1)

    return x.unsqueeze(-1)



# def read_settings(path: str):
#     with open(os.path.join(path, 'settings.yaml')) as f:
#         settings = yaml.load(f, Loader=yaml.FullLoader)
#     return settings