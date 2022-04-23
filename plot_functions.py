from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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


def plot_density_heatmap(xpos, ypos, rho, name, args, combine="sum", save=True, show=True, filename=None):
    ny = 30
    nx = 30
    density_log = np.ones((ny, nx))
    density = np.zeros((ny, nx))
    bins = [[[] for _ in range(nx)] for _ in range(ny)]

    xmin, xmax, ymin, ymax = -1.5, 1.5, -1.5, 1.5
    bin_x = ((xpos - xmin) / ((xmax - xmin) / nx)).astype(np.int32)
    bin_y = ((ypos - ymin) / ((ymax - ymin) / ny)).astype(np.int32)
    bin_x = np.clip(bin_x, 0, nx - 1)
    bin_y = np.clip(bin_y, 0, ny - 1)

    for i in range(bin_x.shape[0]):
        bins[bin_y[i]][bin_x[i]].append(rho[i])
    for i in range(ny):
        for j in range(nx):
            if len(bins[i][j]) > 0:
                if combine == "mean":
                    density_log[i, j] = np.mean(bins[i][j])
                    density[i, j] = np.mean(bins[i][j])
                elif combine == "max":
                    density_log[i, j] = np.max(bins[i][j])
                    density[i, j] = np.max(bins[i][j])
                elif combine == "sum":
                    density_log[i, j] = np.sum(bins[i][j]) #/ xpos.shape[0]
                    density[i, j] = np.sum(bins[i][j]) #/ xpos.shape[0]

    extent = [xmin, xmax, ymin, ymax]
    for i, heatmap in enumerate([np.log(density_log), density]):
        if i == 0:
            continue
        im = plt.imshow(heatmap, extent=extent, origin='lower')
        name_new = ("logRho_" if i == 0 else "Rho_") + name
        plt.title("Density Heatmap for Simulation \n %s" % (name_new))
        plt.xlabel("x-xref")
        plt.ylabel("y-yref")
        plt.colorbar(im, fraction=0.046, pad=0.04, format="%.2f")
        if save:
            if filename is None:
                filename = args.path_plot_densityheat + datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S") + "_heatmap_" + name_new + ".jpg"
            plt.savefig(filename)
        if show:
            plt.show()
        plt.clf()


def plot_losscurves(result, name, args, save=True, show=True, filename=None, type="Loss"):
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
        plt.ylim(0, 0.1)

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
        plt.ylim(0, 1)
    plt.legend()
    plt.xlabel("Episodes")
    plt.tight_layout()
    if save:
        if filename is None:
            filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_%Curve_" % type + name + ".jpg"
        plt.savefig(args.path_plot_loss + filename)
    if show:
        plt.show()
    plt.clf()