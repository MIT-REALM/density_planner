from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot_density_heatmap(xpos, ypos, rho, N, args):
    heatmap, xedges, yedges = np.histogram2d(xpos, ypos, bins=10,
                                             weights=rho)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.title("Density after %.2fs \n(%d simulations, %d states)" % (
        args.dt_sim * N, N, xpos.shape[0]))
    plt.xlabel("x-xref")
    plt.ylabel("y-yref")
    plt.tight_layout()
    filename = args.path_plot_densityheat + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_losscurve.jpg"
    plt.savefig(filename)
    plt.show()


def plot_losscurves(train_loss, test_loss, name, args):
    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.title("Loss Curves for Config \n %s" % (name.replace('weight', '\n weight')))
    plt.xlabel("Episodes")
    plt.ylabel("Loss")
    bottom, top = plt.ylim()
    plt.ylim(bottom, min(0.3, top))
    plt.tight_layout()
    filename = args.path_plot_loss + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_losscurve.jpg"
    plt.savefig(filename)
    plt.show()