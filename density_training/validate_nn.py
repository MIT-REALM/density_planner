from systems.sytem_CAR import Car
import torch
import hyperparams
from plot_functions import plot_scatter
from density_training.utils import load_nn
import numpy as np
from data_generation.utils import load_outputmap, get_input_tensors, get_output_variables



def get_nn_prediction(model, xe0, xref0, t, u_params, args):

    # if args.input_type == "discr10" and u_params.shape[1] < 10:
    #     u_params = torch.cat((u_params[:, :], torch.zeros(u_params.shape[0], 10 - u_params.shape[1])), 1)

    input_tensor, _ = get_input_tensors(u_params.flatten(), xref0, xe0, t * torch.ones(xe0.shape[0]))
    output_map, num_outputs = load_outputmap(xref0.shape[0])

    with torch.no_grad():
        input = input_tensor.to(args.device)
        output = model(input)
        xe, rho = get_output_variables(output, output_map, type='exp')
        rho = rho.unsqueeze(-1).unsqueeze(-1)
    return xe, rho


if __name__ == "__main__":
    sample_size = 100
    args = hyperparams.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = 'finalTrain'
    results = []

    #short_name = run_name
    #nn_name = args.path_nn + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_NN_" + short_name + '.pt'

    # data preparation
    torch.manual_seed(args.random_seed)
    bs = args.batch_size
    system = Car()

    xref_traj, rho_traj, uref_traj, u_params, xe_traj, t = system.get_valid_trajectories(args)
    #xe_LE = x_traj - xref_traj

    # NN prediction
    num_inputs = 30
    model, _ = load_nn(num_inputs, 5, args)
    model.eval()
    step = 5
    t_vec = np.arange(0, args.N_sim * args.dt_sim, step * args.dt_sim)
    for i, t in enumerate(t_vec):
        xe_nn, rho_nn = get_nn_prediction(model, rho_traj[:, 0, 0], x_traj[:, :, 0] - xref_traj[0, :, 0], xref_traj[0, :, 0],
                                          t, uref_traj, num_inputs, args)
        error = xe_nn[:, :, 0] - xe_LE[:, :, i*step]
        print("Max error: %.2f, Mean error: %.2f, MSE: %.2f" %
              (torch.max(torch.abs(error)), torch.mean(torch.abs(error)), torch.mean(torch.abs(error ** 2))))
        plot_scatter(xe_nn[:,:,0], xe_LE[:, :, i*step], rho_nn[:,0,0], rho_traj[:, 0, i*step],
                     run_name + "_time=%.2fs" % t, args, save=True, show=True, filename=None)








