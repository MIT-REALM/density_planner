from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_density_heatmap2(xpos, ypos, rho, name, args, save=True, show=True):
    heatmap, xedges, yedges = np.histogram2d(xpos, ypos, bins=10,
                                             weights=rho)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Density Heatmap for Simulation \n %s" % (name))
    plt.xlabel("x-xref")
    plt.ylabel("y-yref")
    plt.tight_layout()
    if save:
        filename = args.path_plot_densityheat + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_heatmap_" + name + ".jpg"
        plt.savefig(filename)
    if show:
        plt.show()
    plt.clf()


def plot_density_heatmap(xpos, ypos, rho, name, args, combine="mean", save=True, show=True):
    ny = 10
    nx = 10
    density_log = np.ones((ny, nx))
    density = np.zeros((ny, nx))
    bins = [[[] for _ in range(nx)] for _ in range(ny)]

    xmin, xmax, ymin, ymax = -0.5, 0.5, -0.5, 0.5
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
            filename = args.path_plot_densityheat + datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S") + "_heatmap_" + name_new + ".jpg"
            plt.savefig(filename)
        if show:
            plt.show()
        plt.clf()


def plot_losscurves(train_loss, test_loss, name, args, save=True, show=True):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.title("Loss Curves for Config \n %s" % (name))
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    bottom, top = plt.ylim()
    plt.ylim(min(0, max(bottom, -0.1)), min(1000, train_loss[3]))
    plt.tight_layout()
    if save:
        filename = args.path_plot_loss + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_losscurve_" + name + ".jpg"
        plt.savefig(filename)
    if show:
        plt.show()
    plt.clf()