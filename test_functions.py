import torch
from systems.sytem_CAR import Car
from systems.ControlAffineSystem import ControlAffineSystem
import sys
from utils import approximate_derivative, get_grid
import argparse
import matplotlib.pyplot as plt
import numpy as np
import hyperparams


args = hyperparams.parse_args()

torch.manual_seed(args.random_seed)
dt = args.dt_sim
bs = args.batch_size

system = Car()
x = (system.X_MAX - system.X_MIN) * torch.rand(bs, system.DIM_X, 1) + system.X_MIN
uref = (system.UREF_MAX - system.UREF_MIN) * torch.rand(bs, system.DIM_U, 1) + system.UREF_MIN
xref = (system.XREF0_MAX - system.XREF0_MIN) * torch.rand(bs, system.DIM_X, 1) + system.XREF0_MIN

uref_traj = system.sample_uref_traj(args.N_sim)
xref0 = system.sample_xref0()
xref_traj = system.compute_xref_traj(xref0, uref_traj, args.N_sim, args.dt_sim)


def test_dimensions(object: ControlAffineSystem, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor):
    bs = x.shape[0]
    num_state = object.DIM_X
    num_input = object.DIM_U

    u = system.controller(x, xref, uref)
    assert list(u.size()) == [bs, num_input, 1]

    a = system.a_func(x)
    assert list(a.size()) == [bs, num_state, 1]
    b = system.b_func(x)
    assert list(b.size()) == [bs, num_state, num_input]
    f = system.f_func(x, xref, uref)
    assert list(f.size()) == [bs, num_state, 1]

    dadx = system.dadx_func(x)
    assert list(dadx.size()) == [bs, num_state, num_state]
    dbdx = system.dbdx_func(x)
    assert list(dbdx.size()) == [bs, num_state, num_input, num_state]
    dudx = system.dudx_func(x, xref, uref)
    assert list(dudx.size()) == [bs, num_input, num_state]
    dfdx = system.dfdx_func(x, xref, uref)
    assert list(dfdx.size()) == [bs, num_state, num_state]

    divf = system.divf_func(x, xref, uref)
    assert list(divf.size()) == [bs]


def test_derivatives(object: ControlAffineSystem, x: torch.Tensor, xref: torch.Tensor, uref: torch.Tensor):

    dadx = object.dadx_func(x)
    dadx_num = approximate_derivative(object.a_func, x)
    assert torch.all(torch.square(dadx-dadx_num) < 1e-2)

    dbdx = object.dbdx_func(x)
    dbdx_num = approximate_derivative(object.b_func, x)
    assert torch.all(torch.square(dbdx-dbdx_num) < 1e-2)

    dudx = object.dudx_func(x, xref, uref)
    def u_func_helper(x):
        u = object.controller(x, xref, uref)
        return u
    dudx_num = approximate_derivative(u_func_helper, x)
    assert torch.all(torch.square(dudx-dudx_num) < 1e-2)

    dfdx = object.dfdx_func(x, xref, uref)
    def f_func_helper(x):
        f = object.f_func(x, xref, uref)
        return f
    dfdx_num = approximate_derivative(f_func_helper, x)
    assert torch.all(torch.square(dfdx - dfdx_num) < 1e-2)

def test_density(system: ControlAffineSystem, xref_traj, uref_traj, xref0, N_sim, dt):

    params_LE = {
        "sample_size": [15, 15, 5, 5],
        "getDensity": True,
        "title": "LE Density Computation",
    }
    params_MC = {
        "sample_size": [50, 50, 20, 20],
        "getDensity": False,
        "title": "MC Simulation",
    }

    for params in (params_LE, params_MC):
        x = system.sample_x0(xref0, params["sample_size"])

        if params["getDensity"]:
            rho0 = torch.ones(x.shape[0])
            x, rho = system.compute_density(x, xref_traj, uref_traj, rho0, N_sim, dt)
        else:
            rho = torch.ones(x.shape[0])
            with torch.no_grad():
                for i in range(N_sim):
                    x = system.get_next_x(x, xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
        xe = x - xref_traj[[0], :, -1:]

        #scale_rho = 20 / torch.max(rho)
        #plt.scatter(xe[:, 0, 0].numpy(), xe[:, 1, 0].numpy(), s=scale_rho)
        heatmap, xedges, yedges = np.histogram2d(xe[:, 0, 0].numpy(), xe[:, 1, 0].numpy(), bins=10, weights=rho.numpy())
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title(params["title"] + " after %.2fs \n(%d state simulations, random seed %d)" % (
            args.dt_sim * args.N_sim, args.N_sim, args.random_seed))
        plt.xlabel("x-xref")
        plt.ylabel("y-yref")
        plt.show()


test_density(system, xref_traj, uref_traj, xref0, args.N_sim, dt)
test_dimensions(system, x, xref, uref)
test_derivatives(system, x, xref, uref)
print("end")

