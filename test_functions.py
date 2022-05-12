import torch
from systems.sytem_CAR import Car
from systems.ControlAffineSystem import ControlAffineSystem
from systems.utils import approximate_derivative
import hyperparams
from plots.plot_functions import plot_density_heatmap, plot_ref, plot_scatter # plot_density_heatmap2
from density_training.utils import NeuralNetwork, load_nn, get_nn_prediction


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

    params_MC = {
        "sample_size": 20 * 20 * 10 * 10, #[60, 60, 20, 20],
    }
    params_NN = {
        "sample_size": 15 * 15 * 5 * 5,  # [60, 60, 20, 20],
    }
    run_name = "test"

    # MC Estimate
    x0_MC = system.sample_x0(xref0, params_MC["sample_size"])
    rho0_MC = torch.ones(x0_MC.shape[0], 1, 1)
    x_traj_MC, _ = system.compute_density(x0_MC, xref_traj, uref_traj, rho0_MC, xref_traj.shape[2], args.dt_sim, computeDensity=False)
    xe_traj_MC = x_traj_MC - xref_traj

    x0 = system.sample_x0(xref0, params_NN["sample_size"])
    rho0 = torch.ones(x0.shape[0], 1, 1)
    # LE Estimate
    x_traj, rho_traj = system.compute_density(x0, xref_traj, uref_traj, rho0, xref_traj.shape[2], args.dt_sim)
    xe_traj = x_traj - xref_traj

    # NN prediction
    num_inputs = 17
    model, _ = load_nn(num_inputs, 5, args, load_pretrained=True)
    model.eval()
    step = 5
    t_vec = torch.arange(0, args.N_sim * args.dt_sim, step * args.dt_sim)
    for i, t in enumerate(t_vec):
        xe_nn, rho_nn = get_nn_prediction(model, xe_traj[:, :, 0], xref_traj[0, :, 0],
                                          t, u_params, args)
        error = xe_nn[:, :, 0] - xe_traj[:, :, i * step]
        print("Max error: %.3f, Mean error: %.4f" %
              (torch.max(torch.abs(error)), torch.mean(torch.abs(error))))

        plot_density_heatmap(xe_nn[:, :, 0], rho_nn[:, 0, 0], run_name + "_time=%.2fs_NN" % t, args,
                             save=True, show=True, filename=None)
        plot_density_heatmap(xe_traj[:, :, i * step], rho_traj[:, 0, i * step], run_name + "_time=%.2fs_LE" % t, args,
                             save=True, show=True, filename=None)
        plot_density_heatmap(xe_traj_MC[:, :, i * step], 1, run_name + "_time=%.2fs_MC" % t, args,
                             save=True, show=True, filename=None)
        plot_scatter(xe_nn[:50, :, 0], xe_traj[:50, :, i * step], rho_nn[:50, 0, 0], rho_traj[:50, 0, i * step],
                     run_name + "_time=%.2fs" % t, args, save=True, show=True, filename=None)



def test_controller(xref_traj, uref_traj, args):
    x0 = system.sample_x0(xref0, 20)  # get random initial states
    rho0 = torch.ones(x0.shape[0], 1, 1)  # equal initial density
    x_traj, rho_traj = system.compute_density(x0, xref_traj, uref_traj, rho0,
                                              xref_traj.shape[2], args.dt_sim)
    plot_ref(xref_traj, uref_traj, 'test', args, x_traj=x_traj, include_date=True)

if __name__ == "__main__":
    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(8) #(args.random_seed)
    bs = args.batch_size
    system = Car()
    x = (system.X_MAX - system.X_MIN) * torch.rand(bs, system.DIM_X, 1) + system.X_MIN
    xref = (system.XREF0_MAX - system.XREF0_MIN) * torch.rand(bs, system.DIM_X, 1) + system.XREF0_MIN
    uref = (system.UREF_MAX - system.UREF_MIN) * torch.rand(bs, system.DIM_U, 1) + system.UREF_MIN

    while True:
        uref_traj, u_params = system.sample_uref_traj(args)  # get random input trajectory
        xref0 = system.sample_xref0()  # sample random xref
        xref_traj = system.compute_xref_traj(xref0, uref_traj, args)  # compute corresponding xref trajectory
        xref_traj, uref_traj = system.cut_xref_traj(xref_traj,
                                                    uref_traj)  # cut trajectory where state limits are exceeded
        if xref_traj.shape[2] == args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
            break

    test_controller(xref_traj, uref_traj, args)
    test_density(system, xref_traj, uref_traj, xref0, args.N_sim, args.dt_sim)
    test_dimensions(system, x, xref, uref)
    test_derivatives(system, x, xref, uref)

    print("end")

