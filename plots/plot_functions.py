from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from plots.utils import sample_from
import scipy


# def plot_density_heatmap2(xpos, ypos, rho, name, args, save=True, show=True, filename=None):
#     heatmap, xedges, yedges = np.histogram2d(xpos, ypos, bins=10,
#                                              weights=rho)
#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     plt.imshow(heatmap.T, extent=extent, origin='lower')
#     plt.title("Density Heatmap for Simulation \n %s" % (name))
#     plt.xlabel("x-xref")
#     plt.ylabel("y-yref")
#     plt.tight_layout()
#     if save:
#         if filename is None:
#             filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_heatmap_" + name + ".jpg"
#         plt.savefig(args.path_plot_densityheat + filename)
#     if show:
#         plt.show()
#     plt.clf()


def plot_density_heatmap(x, rho, name, args, combine="sampling", save=True, show=True, filename=None, include_date=False):
    """
    Plot a heat map

    Parameters
    ----------
    x: torch.Tensor
        batch_size x self.DIM_X: tensor of states at certain time point
    rho: torch.Tensor
        batch_size: tensor of corresponding densities
    """
    ny = 22
    nx = 22
    sample_size = 1000
    xmin, xmax, ymin, ymax = -1.1, 1.1, -1.1, 1.1

    density_log = np.ones((ny, nx))
    density = np.zeros((ny, nx))
    x = x.detach().cpu().numpy()
    rho = rho.detach().cpu().numpy()

    # get the number of the bin in x and y direction for each state sample
    bin_x = ((x[:,0] - xmin) / ((xmax - xmin) / nx)).astype(np.int32)
    bin_y = ((x[:,1] - ymin) / ((ymax - ymin) / ny)).astype(np.int32)
    bin_x = np.clip(bin_x, 0, nx - 1)
    bin_y = np.clip(bin_y, 0, ny - 1)

    # save density and state 3+4 in list of the corresponding bin
    bins_rho = [[[] for _ in range(nx)] for _ in range(ny)]
    bins_x = [[[] for _ in range(nx)] for _ in range(ny)]
    for i in range(bin_x.shape[0]):
        bins_rho[bin_y[i]][bin_x[i]].append(rho[i])
        bins_x[bin_y[i]][bin_x[i]].append(x[i, 2:])

    for i in range(ny):
        for j in range(nx):
            if len(bins_rho[i][j]) > 0:
                if combine == "mean":
                    density_log[i, j] = np.mean(bins_rho[i][j])
                    density[i, j] = np.mean(bins_rho[i][j])
                elif combine == "max":
                    density_log[i, j] = np.max(bins_rho[i][j])
                    density[i, j] = np.max(bins_rho[i][j])
                elif combine == "sum":
                    density_log[i, j] = np.sum(bins_rho[i][j])  # / xpos.shape[0]
                    density[i, j] = np.sum(bins_rho[i][j])  # / xpos.shape[0]
                elif combine == "sampling" and len(bins_rho[i][j]) > 3:
                    samp_x = np.stack(bins_x[i][j], axis=0).astype(np.double)
                    samp_rho = np.stack(bins_rho[i][j], axis=0).astype(np.double)
                    estimator = LinearNDInterpolator(samp_x, samp_rho, fill_value=0.0)
                    samp_xs, hull1 = sample_from(samp_x, sample_size, sel_indices=[0, 1],
                                                 hull_sampling=True, gain=1, faster_hull=True)
                    dens2 = estimator(*(samp_xs.T))
                    cvh_vol = scipy.spatial.ConvexHull(bins_x[i][j]).volume
                    prob = np.mean(dens2) * cvh_vol
                    density_log[i, j] = prob
                    density[i, j] = prob

    extent = [xmin, xmax, ymin, ymax]
    for i, heatmap in enumerate([np.log(density_log), density]):
        if i == 0:
            continue
        im = plt.imshow(heatmap, extent=extent, origin='lower')
        name_new = ("logRho_" if i == 0 else "Rho_") + name
        plt.title("Heatmap for Simulation \n %s \n Density estimation by %s" % (name_new, combine))
        plt.xlabel("x-xref")
        plt.ylabel("y-yref")
        plt.colorbar(im, fraction=0.046, pad=0.04, format="%.2f")
        plt.tight_layout()
        if save:
            if filename is None:
                if include_date:
                    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_heatmap_" + name_new + ".jpg"
                else:
                    filename = "heatmap_" + name_new + ".jpg"
            plt.savefig(args.path_plot_densityheat + 'random_seed%d/' % args.random_seed + filename)
        if show:
            plt.show()
        plt.clf()


def plot_losscurves(result, name, args, save=True, show=True, filename=None, type="Loss", include_date=False):
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


def plot_scatter(x_nn, x_le, rho_nn, rho_le, name, args, save=True, show=True, filename=None, include_date=False):
    step = int(80 / x_nn.shape[0])
    colors = np.arange(0, step * x_nn.shape[0], step)

    for j in np.arange(0,x_nn.shape[1], 2):
        for i in range(x_nn.shape[0]):
            plt.plot([x_nn[i, j], x_le[i, j]],[x_nn[i, j+1], x_le[i, j+1]], color='gainsboro', zorder=-1)

        plt.scatter(x_nn[:, j], x_nn[:, j+1], marker='o', c=colors, sizes=5 * rho_nn ** (1/6),cmap='gist_ncar',
                    label='NN estimate', zorder=1)
        plt.scatter(x_le[:, j], x_le[:, j+1], marker='x', c=colors, sizes=5 * rho_le ** (1/6), cmap='gist_ncar',
                    label='LE estimate', zorder=1)  # ,

        plt.legend()
        plt.grid()

        if j == 0:
            plt.xlabel("x-xref")
            plt.ylabel("y-yref")
        elif j == 2:
            plt.xlabel("theta-thetaref")
            plt.ylabel("v-vref")

        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        error = x_nn - x_le
        plt.title(name + "\n Max error: %.3f, Mean error: %.4f" %
                  (torch.max(torch.abs(error)), torch.mean(torch.abs(error))))
        plt.tight_layout()
        if save:
            if filename is None:
                if include_date:
                    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_Scatter_" + name + ".jpg"
                else:
                    filename = "Scatter_" + name + ".jpg"
            plt.savefig(args.path_plot_scatter + 'dims%d-%d/' % (j, j+1) + 'random_seed%d/' % args.random_seed + filename)
        if show:
            plt.show()
        plt.clf()