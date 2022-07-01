import os.path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from systems.utils import get_density_map
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from plots.utils import sample_from
import scipy
import matplotlib.colors as pltcol
from motion_planning.utils import pos2gridpos, traj2grid
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_density_heatmap(name, args, xe_dict, rho_dict, save=True, show=True, filename=None,
                         include_date=False, folder=None, log_density=False):
    density_mean = {}
    num_plots = len(xe_dict)
    min_rho = np.inf
    max_rho = 0

    for key in xe_dict:
        density_mean[key], extent = get_density_map(xe_dict[key][:, :2, 0], rho_dict[key][:, 0, 0], args, type=key,
                                                    log_density=log_density)
        mask = density_mean[key] > 0
        if density_mean[key][mask].min() < min_rho:
            min_rho = density_mean[key][mask].min()
        if density_mean[key].max() > max_rho:
            max_rho = density_mean[key].max()

    fig, ax = plt.subplots(1, num_plots)
    #fig.set_figwidth(9.5)
    for i, key in enumerate(xe_dict):
        if num_plots == 1:
            axis = ax
            axis.set_ylabel("y-yref")
        else:
            axis = ax[i]
            ax[0].set_ylabel("y-yref")
        cmap = plt.cm.get_cmap('magma').reversed()
        im = axis.imshow(density_mean[key].T, extent=extent[0]+extent[1], origin='lower', cmap=cmap,
                         norm=pltcol.LogNorm(vmin=min_rho, vmax=max_rho))#vmin=plot_limits[0], vmax=plot_limits[1])
        fig.colorbar(im, ax=axis, orientation='horizontal', format="%.0e")
        axis.set_title("%s Prediction" % (key))
        axis.set_xlabel("x-xref")

    # if num_plots == 2:
    #     fig.suptitle("Density Heatmap at %s                \n max density error: %.3f, mean density error: %.4f            "
    #              % (name, error.max(), error.mean()))
    fig.suptitle("Density Heatmap %s" % name)
    fig.tight_layout()
    #
    # if num_plots == 2:
    #     plt.subplots_adjust(bottom=0.0, right=0.85, top=0.95)
    #cax = plt.axes([0.87, 0.05, 0.02, 0.85])
    #fig.colorbar(im, fraction=0.046, pad=0.04, format="%.0e", cax=cax, shrink=0.6)

    if save:
        if folder is None:
            folder = args.path_plot_densityheat
        if filename is None:
            if include_date:
                filename_new = datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S") + "_heatmap_" + 'randomSeed%d_' % args.random_seed + name + ".jpg"
            else:
                filename_new = "heatmap_" + 'randomSeed%d_' % args.random_seed + name + ".jpg"
            plt.savefig(folder + filename_new)
        else:
            plt.savefig(args.path_plot_densityheat + filename)
    if show:
        plt.show()
    plt.clf()


def plot_scatter(x_nn, x_le, rho_nn, rho_le, name, args, weighted=False,
                 save=True, show=True, filename=None, include_date=False):
    step = int(80 / x_nn.shape[0])
    colors = np.arange(0, step * x_nn.shape[0], step)
    num_plots = 1

    for j in np.arange(0, num_plots*2, 2):
        for i in range(x_nn.shape[0]):
            plt.plot([x_nn[i, j], x_le[i, j]],[x_nn[i, j+1], x_le[i, j+1]], color='gainsboro', zorder=-1)

        if weighted:
            sizes = 5 * rho_nn ** (1/6)
        else:
            sizes = 15 * torch.ones_like(rho_nn)
        plt.scatter(x_nn[:, j], x_nn[:, j+1], marker='o', c=colors, sizes=sizes, cmap='gist_ncar',
                    label='NN estimate', zorder=1)
        plt.scatter(x_le[:, j], x_le[:, j+1], marker='x', c=colors, sizes=sizes, cmap='gist_ncar',
                    label='LE estimate', zorder=1)  # ,
        plt.axis('scaled')
        plt.legend()
        plt.grid()

        if j == 0:
            plt.xlabel("x-xref")
            plt.ylabel("y-yref")
            lim = 2.1
        elif j == 2:
            plt.xlabel("theta-thetaref")
            plt.ylabel("v-vref")
            lim = 1.1
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        ticks_y_grid, ticks_y = plt.yticks()
        plt.xticks(ticks_y_grid[1:-1], ticks_y_grid[1:-1])
        error = x_nn - x_le
        error_dim = torch.sqrt((x_nn[:, j] - x_le[:, j]) ** 2 + (x_nn[:, j+1] - x_le[:, j+1]) ** 2)
        plt.title("State Predictions at " + name + "\n max state error: %.3f, mean state error: %.4f, \n max eucl-distance: %.3f, mean eucl-distance: %.4f" %
                  (torch.max(torch.abs(error)), torch.mean(torch.abs(error)), torch.max(torch.abs(error_dim)), torch.mean(torch.abs(error_dim))))
        plt.tight_layout()
        if save:
            if filename is None:
                if include_date:
                    filename_new = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_Scatter_" + 'dims%d-%d_' % (j, j+1) + 'randomSeed%d_' % args.random_seed + name + ".jpg"
                else:
                    filename_new = "Scatter_" + 'dims%d-%d_' % (j, j+1) + 'randomSeed%d_' % args.random_seed + name + ".jpg"
                plt.savefig(args.path_plot_scatter + filename_new)
            else:
                plt.savefig(args.path_plot_scatter + filename)
        if show:
            plt.show()
        plt.clf()


def plot_losscurves(result, name, args, type="Loss",
                    save=True, show=True, filename=None, include_date=False):
    colors = ['g', 'r', 'c', 'b', 'm', 'y']
    if type == "Loss":
        if "train_loss" in result:
            for i, (key, val) in enumerate((result['train_loss']).items()):
                if not 'max' in key:
                    plt.plot(val, color=colors[i], linestyle='-', label="train " + key)
        if "test_loss" in result:
            for i, (key, val) in enumerate(result['test_loss'].items()):
                if not 'max' in key:
                    plt.plot(val, color=colors[i], linestyle=':', label="test " + key)
        plt.title("Loss Curves for Config \n %s" % (name))
        plt.ylabel("Loss")
        #plt.ylim(0, 0.02)
        plt.yscale('log')

    elif type == "maxError":
        if "train_loss" in result:
            i = 0
            for key, val in (result['train_loss']).items():
                if 'max' in key:
                    if val[0].ndim > 0:
                        for j in range(val[0].shape[0]):
                            plt.plot([val[k][j] for k in range(len(val))],
                                     color=colors[i], linestyle='-', label="train " + key +"[%d]" % j)
                            i += 1
                    else:
                        plt.plot(val, color=colors[i], linestyle='-', label="train " + key)
                        i += 1
        if "test_loss" in result:
            i = 0
            for key, val in (result['train_loss']).items():
                if 'max' in key:
                    if val[0].ndim > 0:
                        for j in range(val[0].shape[0]):
                            plt.plot([val[k][j] for k in range(len(val))],
                                     color=colors[i], linestyle=':', label="test " + key +"[%d]" % j)
                            i += 1
                    else:
                        plt.plot(val, color=colors[i], linestyle=':', label="test " + key)
                        i += 1
        plt.title("Maximum Error for Config \n %s" % (name))
        plt.ylabel("Maximum Error")
        #plt.ylim(0, 10)
        plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.xlabel("Episodes")
    plt.tight_layout()
    if save:
        if filename is None:
            if include_date:
                filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_%Curve_" % type + name + ".jpg"
            else:
                filename = "%Curve_" % type + name + ".jpg"
        plt.savefig(args.path_plot_loss + filename)
    if show:
        plt.show()
    plt.clf()

def plot_cost(costs_dict, args, plot_log=True, name="", save=True, show=True, filename=None, include_date=True, folder=None):
    num_plots = len(costs_dict["cost_sum"][0])
    for i in range(num_plots):
        plt.figure(figsize=(6.4, 4.8))
        plt.title("Cost Curve %d for Motion Planning \n %s" % (i, name))
        for key, val in costs_dict.items():
            if (val[0]).dim() == 0:
                cost = [val[k].item() for k in range(len(val))]
            else:
                cost = [val[k][i].item() for k in range(len(val))]
            if key == "cost_sum":
                plt.plot(cost, linestyle='-', label="sum")
            else:
                plt.plot(cost, linestyle=':', label=key)
        plt.ylabel("Costs")
        if plot_log:
            plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.xlabel("Episodes")
        plt.tight_layout()
        if save:
            if folder is None:
                folder = args.path_plot_cost
            if filename is None:
                if include_date:
                    fname = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_Curve%d_" % i + name + ".jpg"
                else:
                    fname = "Curve%d_" % i + name + ".jpg"
            else:
                fname = filename
            plt.savefig(folder + fname)
            plt.close()
        if show:
            plt.show()
        plt.clf()

def plot_ref(xref_traj, uref_traj, name, args, system, t=None, x_traj=None,
             save=True, show=True, filename=None, include_date=False, folder=None):
    # xref_traj[0, 2, :] = (xref_traj[0, 2, :] + np.pi) % (2 * np.pi) - np.pi
    # x_traj[:, 2, :] = (x_traj[:, 2, :] + np.pi) % (2 * np.pi) - np.pi
    #xref_traj = system.project_angle(xref_traj)
    if t is None:
        t = args.dt_sim * torch.arange(0, xref_traj.shape[2])
    fig, ax = plt.subplots(5, 1, gridspec_kw={'height_ratios': [4, 1, 1, 1, 1]})
    fig.set_figheight(13)

    if x_traj is not None:
        #x_traj = system.project_angle(x_traj)
        for i in range(x_traj.shape[0]):
            if i == 1:
                ax[0].plot(x_traj[i, 0, :], x_traj[i, 1, :], 'slategrey', label='Sample Trajectories')
            else:
                ax[0].plot(x_traj[i, 0, :], x_traj[i, 1, :], 'slategrey')
            tracking_error = torch.sqrt((x_traj[i, 0, :]-xref_traj[0,0,:]) ** 2 + (x_traj[i, 1, :]-xref_traj[0,1,:]) ** 2)
            ax[1].plot(t, tracking_error, 'slategrey')
            ax[2].plot(t, x_traj[i, 2, :], 'slategrey')
            ax[3].plot(t, x_traj[0, 3, :], 'slategrey')

    ax[0].plot(xref_traj[0,0,:], xref_traj[0,1,:], 'firebrick', label='Reference Trajectory')
    ax[0].grid()
    ax[0].set_xlabel("x-position")
    ax[0].set_ylabel("y-position")
    ax[0].set_xlim(system.X_MIN[0, 0, 0] - 0.1, system.X_MAX[0, 0, 0] + 0.1)
    ax[0].set_ylim(system.X_MIN[0, 1, 0] - 0.1, system.X_MAX[0, 1, 0] + 0.1)
    ax[0].legend()
    ax[0].set_title("Reference Trajectories of type " + args.input_type + "\n" + name)

    ax[1].plot(t, torch.zeros_like(t), 'firebrick')
    ax[1].grid()
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Tracking Error")
    ax[1].set_ylim(0, torch.sqrt(system.XE0_MAX[0, 0, 0] ** 2 + system.XE0_MAX[0, 1, 0] ** 2) + 1)

    ax[2].plot(t, xref_traj[0, 2, :], 'firebrick')
    ax[2].grid()
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Heading angle")
    ax[2].set_ylim(system.X_MIN[0, 2, 0] - 0.1, system.X_MAX[0, 2, 0] + 0.1)

    ax[3].plot(t, xref_traj[0, 3, :], 'firebrick')
    ax[3].grid()
    ax[3].set_xlabel("Time")
    ax[3].set_ylabel("Velocity")
    ax[3].set_ylim(system.X_MIN[0, 3, 0] - 0.1, system.X_MAX[0, 3, 0] + 0.1)

    ax[4].plot(t[:uref_traj.shape[2]], uref_traj[0, 0, :], label='Angular velocity')
    ax[4].plot(t[:uref_traj.shape[2]], uref_traj[0, 1, :], label='Longitudinal acceleration')
    ax[4].grid()
    ax[4].set_xlabel("Time")
    ax[4].set_ylabel("Reference Inputs")
    ax[4].set_ylim(system.UREF_MIN[0, :, 0].min() - 0.1, system.UREF_MAX[0, :, 0].max() + 0.1)
    ax[4].legend()


    fig.tight_layout()
    if save:
        if folder is None:
            folder = args.path_plot_references
        if filename is None:
            if include_date:
                filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_References_" + name + ".jpg"
            else:
                filename = "References_" + name + ".jpg"
        plt.savefig(folder + filename)
    if show:
        plt.show()
    plt.clf()


def plot_grid(object, args, timestep=None, cmap='binary', name=None,
              save=True, show=True, filename=None, include_date=False, folder=None):
    if save:
        plt.close("all")
    if torch.is_tensor(object):
        if object.dim() == 3:
            if timestep is None:
                grid = object[:, :, -1]
                str_timestep = f"\n at timestep={object.shape[2]-1}"
            else:
                grid = object[:, :, timestep]
                str_timestep = f"\n at timestep={timestep}"
        else:
            grid = object
            str_timestep = ""
        if name is None:
            name = "grid"
    else:
        if timestep is None:
            timestep = object.current_timestep
        str_timestep = f"\n at timestep={timestep}"
        grid = object.grid[:, :, timestep]
        if name is None:
            name = object.name
    x_wide = max((args.environment_size[1] - args.environment_size[0]) / 10, 3)
    y_wide = (args.environment_size[3] - args.environment_size[2]) / 10
    plt.figure(figsize=(x_wide, y_wide), dpi=200)
    plt.pcolormesh(grid.T, cmap=cmap, norm=None)
    plt.axis('scaled')

    ticks_x = np.concatenate((np.arange(0, args.environment_size[1]+1, 10), np.arange(-10, args.environment_size[0]-1, -10)), 0)
    ticks_y = np.concatenate((np.arange(0, args.environment_size[3]+1, 10), np.arange(-10, args.environment_size[2]-1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)

    plt.title(f"{name}" + str_timestep)
    plt.tight_layout()
    if save:
        if folder is None:
            folder = args.path_plot_grid
        if filename is None:
            if include_date:
                filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_Grid_" + name + ".jpg"
            else:
                filename = "Grid_" + name + ".jpg"
        plt.savefig(folder + filename, dpi=200)
    else:
        plt.show()
    plt.clf()



# def plot_density_heatmap2(x, rho, name, args, plot_limits=None, combine="mean", save=True, show=True, filename=None, include_date=False):
#     """
#     Plot a heat map
#
#     Parameters
#     ----------
#     x: torch.Tensor
#         batch_size x self.DIM_X: tensor of states at certain time point
#     rho: torch.Tensor
#         batch_size: tensor of corresponding densities
#     """
#
#     if plot_limits is None:
#         plot_limits = [-np.inf, np.inf]
#     ny = 22
#     nx = 22
#     sample_size = 1000
#     xmin, xmax, ymin, ymax = -2.1, 2.1, -2.1, 2.1
#
#     density_log = np.ones((ny, nx))
#     density = np.zeros((ny, nx))
#     x = x.detach().cpu().numpy()
#     rho = rho.detach().cpu().numpy()
#
#     # get the number of the bin in x and y direction for each state sample
#     bin_x = ((x[:,0] - xmin) / ((xmax - xmin) / nx)).astype(np.int32)
#     bin_y = ((x[:,1] - ymin) / ((ymax - ymin) / ny)).astype(np.int32)
#     bin_x = np.clip(bin_x, 0, nx - 1)
#     bin_y = np.clip(bin_y, 0, ny - 1)
#
#     # save density and state 3+4 in list of the corresponding bin
#     bins_rho = [[[] for _ in range(nx)] for _ in range(ny)]
#     bins_x = [[[] for _ in range(nx)] for _ in range(ny)]
#     for i in range(bin_x.shape[0]):
#         bins_rho[bin_y[i]][bin_x[i]].append(rho[i])
#         bins_x[bin_y[i]][bin_x[i]].append(x[i, 2:])
#
#     for i in range(ny):
#         for j in range(nx):
#             if len(bins_rho[i][j]) > 0:
#                 if combine == "mean":
#                     density_log[i, j] = np.mean(bins_rho[i][j])
#                     density[i, j] = np.mean(bins_rho[i][j])
#                 elif combine == "max":
#                     density_log[i, j] = np.max(bins_rho[i][j])
#                     density[i, j] = np.max(bins_rho[i][j])
#                 elif combine == "sum":
#                     density_log[i, j] = np.sum(bins_rho[i][j])  # / xpos.shape[0]
#                     density[i, j] = np.sum(bins_rho[i][j])  # / xpos.shape[0]
#                 elif combine == "sampling" and len(bins_rho[i][j]) > 3:
#                     samp_x = np.stack(bins_x[i][j], axis=0).astype(np.double)
#                     samp_rho = np.stack(bins_rho[i][j], axis=0).astype(np.double)
#                     estimator = LinearNDInterpolator(samp_x, samp_rho, fill_value=0.0)
#                     samp_xs, hull1 = sample_from(samp_x, sample_size, sel_indices=[0, 1],
#                                                  hull_sampling=True, gain=1, faster_hull=True)
#                     dens2 = estimator(*(samp_xs.T))
#                     cvh_vol = scipy.spatial.ConvexHull(bins_x[i][j]).volume
#                     prob = np.mean(dens2) * cvh_vol
#                     density_log[i, j] = prob
#                     density[i, j] = prob
#
#     extent = [xmin, xmax, ymin, ymax]
#     for i, heatmap in enumerate([np.log(density_log), density]):
#         if i == 0:
#             continue
#         im = plt.imshow(heatmap, extent=extent, origin='lower', vmin=plot_limits[0], vmax=plot_limits[1])
#         name_new = ("logRho_" if i == 0 else "Rho_") + name
#         plt.title("Heatmap for Simulation \n %s \n Density estimation by %s" % (name_new, combine))
#         plt.xlabel("x-xref")
#         plt.ylabel("y-yref")
#         plt.colorbar(im, fraction=0.046, pad=0.04, format="%.2f")
#         plt.tight_layout()
#         if save:
#             if filename is None:
#                 if include_date:
#                     filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_heatmap_" + 'randomSeed%d_' % args.random_seed + name_new + ".jpg"
#                 else:
#                     filename = "heatmap_" + 'randomSeed%d_' % args.random_seed + name_new + ".jpg"
#             plt.savefig(args.path_plot_densityheat + filename)
#         if show:
#             plt.show()
#         plt.clf()