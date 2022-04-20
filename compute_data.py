import torch
from systems.sytem_CAR import Car
import hyperparams
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle


random_seed = False
plot = False
sample_size = 1000 # [15, 15, 5, 5]
iteration_number = 1000
system = Car()

args = hyperparams.parse_args()
if random_seed:
    torch.manual_seed(args.random_seed)
else:
    args.random_seed = None

results_all = []
for i in range(iteration_number):

    # get random input trajectory and compute corresponding state trajectory
    uref_traj = system.sample_uref_traj(args.N_sim, args.N_u)
    xref0 = system.sample_xref0()
    xref_traj = system.compute_xref_traj(xref0, uref_traj, args.N_sim, args.dt_sim)
    xref_traj, uref_traj = system.cut_xref_traj(xref_traj, uref_traj) # cut trajectory where state limits are exceeded
    if xref_traj.shape[2] < args.N_sim:
        continue

    x0 = system.sample_x0(xref0, sample_size)
    rho0 = 1 / (torch.prod(system.X_MAX-system.X_MIN)) * torch.ones(x0.shape[0], 1, 1)
    x_traj, rho_traj = system.compute_density(x0, xref_traj, uref_traj, rho0, xref_traj.shape[2], args.dt_sim)
    if x_traj.shape[2] < args.N_sim:
        continue

    xref_traj = xref_traj[[0],:, :x_traj.shape[2]]
    uref_traj = uref_traj[[0],:, :x_traj.shape[2]]
    xe_traj = x_traj - xref_traj
    t = args.dt_sim * torch.arange(0, x_traj.shape[2])
    results = {
        't': t,
        'uref_traj': uref_traj,
        'xref_traj': xref_traj,
        'xe_traj': xe_traj,
        'rho_traj': rho_traj
    }

    if plot:
        heatmap, xedges, yedges = np.histogram2d(xe_traj[:, 0, -1].numpy(), xe_traj[:, 1, -1].numpy(), bins=10, weights=rho_traj[:,0,-1].numpy())
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.imshow(heatmap.T, extent=extent, origin='lower')
        plt.title("Density after %.2fs \n(%d simulations, %d states)" % (
            args.dt_sim * x_traj.shape[2], x_traj.shape[2], x_traj.shape[0]))
        plt.xlabel("x-xref")
        plt.ylabel("y-yref")
        plt.show()

    results_all.append(results)

path = args.path_rawdata
data_name = path+datetime.now().strftime("%Y-%m-%d-%H-%M-%S")+"_"+system.systemname+"_dt%d" % (args.dt_sim*1000)+"ms_Nsim%d_Nu%d_iter%d" % (args.N_sim, args.N_u, iteration_number)+".pickle" #+"_randomSeed"+str(args.random_seed)
with open(data_name, "wb") as f:
    pickle.dump(results_all, f)
print("end")