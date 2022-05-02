import torch
from systems.sytem_CAR import Car
from systems.ControlAffineSystem import ControlAffineSystem
from systems.utils import approximate_derivative
import hyperparams
from plots.plot_functions import plot_density_heatmap # plot_density_heatmap2
from density_training.train_density import NeuralNetwork


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
        "sample_size": 15 * 15 * 5 * 5, #15, 15, 5, 5],
        "title": "LE",
    }
    params_MC = {
        "sample_size": 60 * 60 * 20 * 20, #[60, 60, 20, 20],
        "title": "MC",
    }
    params_NN = {
        "sample_size": 15 * 15 * 5 * 5,  # [60, 60, 20, 20],
        "title": "NN",
        "filename": "2022-04-23-17-07-52_NN_lrStepEvery100BS512layer3noreg_lr0.001_numHidd4_sizeHidd64_rhoLoss0.1"
    }
    x0 = system.sample_x0(xref0, params_NN["sample_size"])
    rho = torch.ones(x0.shape[0], 1, 1)

    for params in (params_NN,params_LE, params_MC): #
        x = x0
        if params["title"] == "LE":
            x, rho = system.compute_density(x, xref_traj, uref_traj, rho, N_sim, dt, cutting=False)
            comb = "sum"
            xe = x - xref_traj
        elif params["title"] == "MC":
            with torch.no_grad():
                for i in range(N_sim):
                    x = system.get_next_x(x, xref_traj[:, :, [i]], uref_traj[:, :, [i]], dt)
            comb = "sum"
            xe = x - xref_traj
        elif params["title"] == "NN":
            model_params, args_NN, result = torch.load(('data/trained_nn/' + params["filename"] + '.pt'), map_location=args.device)
            num_inputs = 30
            model = NeuralNetwork(num_inputs-1, 5, args_NN).to(args.device)
            model.load_state_dict(model_params)
            model.eval()
            u_in = uref_traj[0, :, ::args_NN.N_u]
            if u_in.shape[1] < 10:
                u_in = torch.cat((u_in[:, :], torch.zeros(u_in.shape[0], 10 - u_in.shape[1])), 1)
            input_map = {'rho0': 0,
                         'xe0': torch.arange(1, xref_traj.shape[1] + 1),
                         'xref0': torch.arange(xref_traj.shape[1] + 1, 2 * xref_traj.shape[1] + 1),
                         't': 2 * xref_traj.shape[1] + 1,
                         'uref': torch.arange(2 * xref_traj.shape[1] + 2, num_inputs)}
            input_tensor = torch.zeros(x.shape[0], num_inputs)
            input_tensor[:, input_map['uref']] = (torch.cat((u_in[0, :], u_in[1, :]), 0)).repeat(x.shape[0],1)
            input_tensor[:, input_map['xe0']] = x[:, :, 0] - xref_traj[0, :, 0]
            input_tensor[:, input_map['rho0']] = rho[:, 0, 0]
            input_tensor[:, input_map['t']] = torch.ones(x.shape[0]) * N_sim * args.dt_sim
            input_tensor[:, input_map['xref0']] = xref_traj[0, :, 0]
            with torch.no_grad():
                input = input_tensor.to(args.device)
                output = model(input[:, 1:])
                xe = output[:, 0:4].unsqueeze(-1)
                rho = input[:, 0] * torch.exp(output[:, 4])
                rho = rho.unsqueeze(-1).unsqueeze(-1)
            comb = "mean"



        name = params["title"] + "_time%.2fs_numSim%d_numStates%d_randSeed%d" % (
             args.dt_sim * args.N_sim, args.N_sim, xe.shape[0], args.random_seed)
        #plot_density_heatmap2(xe[:, 0, -1].numpy(), xe[:, 1, -1].numpy(), rho[:, 0, -1].numpy(), name, args, save=False)
        plot_density_heatmap(xe[:, 0, -1].numpy(), xe[:, 1, -1].numpy(), rho[:, 0, -1].numpy(), name, args, combine=comb, save=False)

if __name__ == "__main__":
    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.random_seed)
    bs = args.batch_size
    system = Car()
    x = (system.X_MAX - system.X_MIN) * torch.rand(bs, system.DIM_X, 1) + system.X_MIN
    xref = (system.XREF0_MAX - system.XREF0_MIN) * torch.rand(bs, system.DIM_X, 1) + system.XREF0_MIN

    while True:
        uref_traj = system.sample_uref_traj(args.N_sim, args.N_u)  # get random input trajectory
        xref0 = system.sample_xref0()  # sample random xref
        xref_traj = system.compute_xref_traj(xref0, uref_traj, args.N_sim,
                                             args.dt_sim)  # compute corresponding xref trajectory
        xref_traj, uref_traj = system.cut_xref_traj(xref_traj,
                                                    uref_traj)  # cut trajectory where state limits are exceeded
        if xref_traj.shape[2] == args.N_sim:  # start again if reference trajectory is shorter than 0.9 * N_sim
            break

    test_density(system, xref_traj, uref_traj, xref0, args.N_sim, args.dt_sim)
    test_dimensions(system, x, xref, uref)
    test_derivatives(system, x, xref, uref)

    print("end")

