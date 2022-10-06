import torch
from systems.sytem_CAR import Car
import hyperparams
from datetime import datetime
import pickle
from plots.plot_functions import plot_density_heatmap


def compute_data(iteration_number, samples_x, system, args,samples_t=None, save=True, plot=True):
    """
    function to create the data for training the neural density predictor

    :param iteration_number:    list with number of files which will be generated and size of each file
    :param samples_x:           number of sampled initial conditions x0
    :param system:              system which will be used
    :param args:                settings
    :param samples_t:           number of time points which will be saved
    :param save:                True if files should be saved
    :param plot:                True if heatmap should be plotted
    :return: results_all        data which was generated
    """
    results_all = []

    for j in range(iteration_number[0]):
        for i in range(iteration_number[1]):
            xref_traj, rholog_traj, uref_traj, u_params, xe_traj, t = system.get_valid_trajectories(samples_x, args)
            if samples_t == 0:
                indizes = args.N_sim - 1
            elif samples_t is not None:
                if False:  # t[-1] > 3: # additional samples at the beginning of trajectory
                    indizes = torch.randint(0, t.shape[0], (int(0.5 * samples_t),))
                    indizes = torch.cat((indizes, torch.randint(0, int(2 / t[1]), (int(0.5 * samples_t),))), 0)
                else:
                    indizes = torch.randint(0, t.shape[0], (int(samples_t),))
                indizes = list(set(indizes.tolist()))
            else:
                indizes = torch.arange(0, t.shape[0])
            results = {
                'u_params': u_params,
                'xe0': xe_traj[:, :, 0],
                't': t[indizes],
                'xref0': xref_traj[0, :, 0],
                'xe_traj': xe_traj[:, :, indizes],
                'rholog_traj': rholog_traj[:, :, indizes]
            }
            results_all.append(results)

            if plot:
                name = "LE_time%.2fs_numSim%d_numStates%d)" % (
                    args.dt_sim * xe_traj.shape[2], xe_traj.shape[2], xe_traj.shape[0])
                plot_density_heatmap(xe_traj[:, 0, -1].numpy(), xe_traj[:, 1, -1].numpy(), rho_traj[:, 0, -1].numpy(), name,
                                     args)
        if save:
            path = args.path_rawdata
            data_name = path + datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S") + "_rawData_" + system.systemname + "_%s_Nsim%d_iter%d_xSamples%d_tSamples%d" % (
                        args.input_type, args.N_sim, iteration_number[1], samples_x, samples_t) + ".pickle"
            print("save " + data_name)
            with open(data_name, "wb") as f:
                pickle.dump([results_all, args], f)
            results_all = []

    return results_all



if __name__ == "__main__":

    args = hyperparams.parse_args()

    samples_x = args.samplesX_rawdata  # number of sampled initial conditions x0
    samples_t = 10  # number of random time points
    iteration_number = [args.iterations_rawdata, args.size_rawdata] # number of files, size of each file
    system = Car(args)  # system which will be used

    random_seed = False
    print("Starting data generation")
    if random_seed:
        torch.manual_seed(args.random_seed)
    else:
        args.random_seed = None

    compute_data(iteration_number, samples_x, system, args, samples_t=samples_t, save=True, plot=False)

    print("end")
