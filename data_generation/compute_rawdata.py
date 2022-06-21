import torch
from systems.sytem_CAR import Car
import hyperparams
from datetime import datetime
import pickle
from plots.plot_functions import plot_density_heatmap, plot_ref
from multiprocessing.pool import Pool
import time
import os

def compute_data(iteration_number, samples_x, system, args,samples_t=None, save=True, plot=True):
    results_all = []

    for j in range(iteration_number[0]):
        for i in range(iteration_number[1]):
            xref_traj, rho_traj, uref_traj, u_params, xe_traj, t = system.get_valid_trajectories(samples_x, args)
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
                # 'uref_traj': uref_traj,
                'u_params': u_params,
                'xe0': xe_traj[:, :, 0],
                't': t[indizes],
                'xref0': xref_traj[0, :, 0],
                'xref_traj': xref_traj[:, :, indizes],
                'xe_traj': xe_traj[:, :, indizes],
                'rho_traj': rho_traj[:, :, indizes]
            }
            results_all.append(results)

            if plot:
                name = "LE_time%.2fs_numSim%d_numStates%d)" % (
                    args.dt_sim * xe_traj.shape[2], xe_traj.shape[2], xe_traj.shape[0])
                plot_density_heatmap(xe_traj[:, 0, -1].numpy(), xe_traj[:, 1, -1].numpy(), rho_traj[:, 0, -1].numpy(), name,
                                     args)
        # print("Finished in %.4f s" % (time.time() - t1))
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

def worker_job(input_data):
    iteration_number, samples_x, system, args, samples_t = input_data
    compute_data(iteration_number, samples_x, system, args, samples_t=samples_t, save=True, plot=False)

if __name__ == "__main__":

    args = hyperparams.parse_args()
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    samples_x = args.samplesX_rawdata  # [15, 15, 5, 5] ' number of sampled initial conditions x0
    samples_t = args.samplesT_rawdata
    iteration_number = [args.iterations_rawdata, args.size_rawdata]
    system = Car()
    random_seed = False
    print("Starting data generation")
    if random_seed:
        torch.manual_seed(args.random_seed)
    else:
        args.random_seed = None

    if args.num_jobs == 1:
        compute_data(iteration_number, samples_x, system, args, samples_t=samples_t, save=True, plot=False)

    else:
        input_list = []
        for i in range(args.num_jobs):
            input_list.append((iteration_number, samples_x, system, args, samples_t))
        pool = Pool(processes=args.num_workers)
        outputs = pool.map(worker_job, input_list)


    compute_data(iteration_number, samples_x, system, args, samples_t=samples_t, save=True, plot=False)
    #compute_data(iteration_number, 100, system, args, samples_t=0, save=True,
    #               plot=False) #samples_t=0... no sample times inbetween, just final value after N_sim-1 timesteps saved


    print("end")
