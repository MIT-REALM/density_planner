from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from systems.utils import get_density_map
import matplotlib
import matplotlib.colors as pltcol
from motion_planning.utils import pos2gridpos, traj2grid, pred2grid, convert_color

### settings
plt.style.use('seaborn-paper')
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{{mathrsfs}}'
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rcParams['legend.fontsize'] = 18
plt.rc('axes', titlesize=18)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

###colors
MITRed = (163/256, 31/256, 52/256)
TUMBlue = "#0065BD"
TUMBlue_med = (153/256, 193/256, 229/256)
TUMBlue_light = (230/256, 240/256, 249/256)
TUMGray = "#808080"
TUMGray_light = "#CCCCC6"
TUMOrange_acc = "#E37222"
TUMGreen_acc = "#A2AD00"


"""
functions to generate plots and figures
"""

def plot_density_heatmap(name, args, xe_dict, rho_dict, system=None, save=True, show=True, filename=None,
                         include_date=False, folder=None, log_density=False):
    """
    function for plotting the density and state predictions in a heatmap
    """

    plt.rcParams['legend.fontsize'] = 20
    plt.rc('axes', titlesize=20)
    plt.rc('axes', labelsize=20)
    plt.rc('xtick', labelsize=18)
    plt.rc('ytick', labelsize=18)

    density_mean = {}
    num_plots = len(xe_dict)
    min_rho = np.inf
    max_rho = 0

    for key in xe_dict:
        density_mean[key], extent = get_density_map(xe_dict[key][:, :2, 0], rho_dict[key][:, 0, 0], args, type=key,
                                                    log_density=log_density, system=system)
        mask = density_mean[key] > 0
        if density_mean[key][mask].min() < min_rho:
            min_rho = density_mean[key][mask].min()
        if density_mean[key].max() > max_rho:
            max_rho = density_mean[key].max()

    fig, ax = plt.subplots(num_plots + 1, 1, gridspec_kw={'height_ratios': [1] * num_plots + [0.1]})
    fig.set_figheight(9.5)
    fig.set_figwidth(4)
    for i, key in enumerate(xe_dict):
        axis = ax[i]
        cmap = plt.cm.get_cmap('magma').reversed()
        if key == "FPE":
            im = axis.imshow(density_mean[key].T, extent=extent[0] + extent[1], origin='lower', cmap=cmap)
        else:
            im = axis.imshow(density_mean[key].T, extent=extent[0] + extent[1], origin='lower', cmap=cmap,
                         norm=pltcol.LogNorm(vmin=min_rho, vmax=max_rho))#vmin=plot_limits[0], vmax=plot_limits[1])
        axis.set_title("\\textbf{%s Prediction}" % (key))
        axis.set_xlabel("$p_x-p_{x*}$ [m]")
        axis.set_ylabel("$p_y-p_{y*}$ [m]")

    fig.colorbar(im, ax=ax[num_plots], orientation='horizontal', fraction=0.8, pad=0.2, format="%.0e")
    ax[num_plots].set_title("\\textbf{Density Scale}")
    ax[num_plots].axis('off')
    fig.tight_layout()

    if save:
        if folder is None:
            folder = args.path_plot_densityheat
        if filename is None:
            if include_date:
                filename_new = datetime.now().strftime(
                    "%Y-%m-%d-%H-%M-%S") + "_heatmap_" + 'randomSeed%d_' % args.random_seed + name + ".pdf"
            else:
                filename_new = "heatmap_" + 'randomSeed%d_' % args.random_seed + name + ".pdf"
            plt.savefig(folder + filename_new)
        else:
            plt.savefig(args.path_plot_densityheat + filename)
    if show:
        plt.show()
    plt.clf()


def plot_scatter(x_nn, x_le, rho_nn, rho_le, name, args, weighted=False,
                 save=True, show=True, filename=None, include_date=False):
    """
    function for plotting the predicted and the true states in a scatter plot
    """

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
    """
    function for plotting the loss of the density NN
    """

    colors = [MITRed, TUMBlue, TUMGray] #['g', 'r', 'c', 'b', 'm', 'y']
    plt.figure(figsize=(8, 4.5))
    if type == "Loss":
        if "train_loss" in result:
            for i, (key, val) in enumerate((result['train_loss']).items()):
                if not 'max' in key:
                    if key == "loss":
                        label = "$\mathscr{L}~$"
                    elif key == "loss_xe":
                        label = "$\mathscr{L}_\mathbf{x}$"
                    elif key == "loss_rho_w":
                        label = "$\mathscr{L}_g$"
                    else:
                        label = key
                    plt.plot(val, color=colors[i], linestyle='-', label=label + " (training)")
        if "test_loss" in result:
            for i, (key, val) in enumerate(result['test_loss'].items()):
                if not 'max' in key:
                    if key == "loss":
                        label = "$\mathscr{L}~$"
                    elif key == "loss_xe":
                        label = "$\mathscr{L}_\mathbf{x}$"
                    elif key == "loss_rho_w":
                        label = "$\mathscr{L}_g$"
                    else:
                        label = key
                    plt.plot(val, color=colors[i], linestyle=':', label=label + " (test)")
        # plt.title("Loss Curves for Config \n %s" % (name))
        plt.ylabel("Loss")
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
        plt.yscale('log')
    plt.legend(ncol=2, loc='upper right')
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
    """
    function for plotting the cost curves of the gradient-based trajectory optimization method
    """
    num_plots = len(costs_dict["cost_sum"][0])
    for i in range(num_plots):
        plt.figure(figsize=(8, 5))
        #plt.title("Cost Curve %d for Motion Planning \n %s" % (i, name))
        for key in ["cost_goal", "cost_uref", "cost_bounds", "cost_coll", "cost_sum"]:
            val = costs_dict[key]
            if (val[0]).dim() == 0:
                cost = [val[k].item() for k in range(len(val))]
            else:
                cost = [val[k][i].item() for k in range(len(val))]
            if key == "cost_sum":
                plt.plot(cost, linestyle='-', label="Total cost $J$", color=MITRed)
            else:
                if key == "cost_goal":
                    label = "$\\alpha_{\\textrm{\\LARGE{goal}}} ~J_{\\textrm{\\LARGE{goal}}}$"
                    col = TUMBlue
                elif key == "cost_uref":
                    label = "$\\alpha_{\\textrm{\\LARGE{input}}} ~J_{\\textrm{\\LARGE{input}}}$"
                    col = TUMGray_light
                elif key == "cost_bounds":
                    label = "$\\alpha_{\\textrm{\\LARGE{bounds}}} ~J_{\\textrm{\\LARGE{bounds}}}$"
                    col = TUMOrange_acc
                elif key == "cost_coll":
                    label = "$\\alpha_{\\textrm{\\LARGE{coll}}} ~J_{{\\textrm{\\LARGE{coll}}}}$"
                    col = TUMGreen_acc
                plt.plot(cost, linestyle=':', label=label, color=col)
        plt.ylabel("Cost")
        if plot_log:
            plt.yscale('log')
        plt.legend(bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=3)
        plt.grid()
        plt.xlabel("Iterations")
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
    """
    function for plotting the reference trajectory and the resulting state trajectories
    """
    if t is None:
        t = args.dt_sim * torch.arange(0, xref_traj.shape[2])
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]})
    fig.set_figheight(8) # for 5 plots: 13
    matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'

    if x_traj is not None:
        #x_traj = system.project_angle(x_traj)
        for i in range(x_traj.shape[0]):
            if i == 1:
                ax[0].plot(x_traj[i, 0, :], x_traj[i, 1, :], color=TUMGray_light,
                           label='Sample Trajectories $\{\mathbf{x}^{(i)}(\cdot)\}_{i=1}^{50}$')
            else:
                ax[0].plot(x_traj[i, 0, :], x_traj[i, 1, :], color=TUMGray_light)
            tracking_error = torch.sqrt((x_traj[i, 0, :]-xref_traj[0,0,:]) ** 2 + (x_traj[i, 1, :]-xref_traj[0,1,:]) ** 2)
            ax[1].plot(t, tracking_error, color=TUMGray_light)

    ax[0].plot(xref_traj[0,0,:], xref_traj[0,1,:],  color=TUMBlue,
               label='Reference Trajectory $\mathbf{x}_*(\cdot)$')
    ax[0].grid()
    ax[0].set_xlabel("$p_x$ [m]")
    ax[0].set_ylabel("$p_y$ [m]")
    ax[0].set_xlim(system.X_MIN[0, 0, 0] - 1, system.X_MAX[0, 0, 0] + 1)
    ax[0].set_ylim(system.X_MIN[0, 1, 0] - 1, system.X_MAX[0, 1, 0] + 1)
    ax[0].set_xticks(np.arange(system.X_MIN[0, 0, 0], system.X_MAX[0, 0, 0]+1, 5))
    ax[0].set_yticks(np.arange(system.X_MIN[0, 1, 0], system.X_MAX[0, 1, 0]+1, 5))
    ax[0].legend()

    ax[1].plot(t, torch.zeros_like(t), color=TUMBlue)
    ax[1].grid()
    ax[1].set_xlabel("$t$ [s]")
    ax[1].set_ylabel("$\left\Vert\\begin{bmatrix} p_x - p_{x*} \\\\ p_y -p_{y*} \end{bmatrix}\\right\Vert~$ [m]")
    ax[1].set_ylim(0, torch.sqrt(system.XE0_MAX[0, 0, 0] ** 2 + system.XE0_MAX[0, 1, 0] ** 2) + 1)
    ax[1].set_title("$\quad$ ")

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
    """
    function for occupation map of the environment
    """

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
    x_wide = max(np.abs((args.environment_size[1] - args.environment_size[0])) / 10, 3)
    y_wide = np.abs((args.environment_size[3] - args.environment_size[2])) / 10
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


def plot_motion(i, cmap, x_traj, rho_traj, xref_traj, args, grid_env_sc):
    """
    function for plotting the position of the sample trajectories weighted by their density at a certain time point in
    the occupation map
    """

    with torch.no_grad():
        # 3. compute marginalized density grid
        grid_pred = pred2grid(x_traj[:, :, [i]], rho_traj[:, :, [i]], args, return_gridpos=False)

    grid_pred_sc = 127 * torch.clamp(grid_pred / grid_pred.max(), 0, 1)
    grid_pred_sc[grid_pred_sc != 0] += 128
    grid_traj = traj2grid(xref_traj[:, :, :i + 1], args)
    grid_traj[grid_traj != 0] = 256
    grid_all = torch.clamp(grid_env_sc[:, :, i] + grid_traj + grid_pred_sc[:, :, 0], 0, 256)

    x_wide = max(np.abs((args.environment_size[1] - args.environment_size[0])) / 10, 3)
    y_wide = np.abs((args.environment_size[3] - args.environment_size[2])) / 10
    plt.figure(figsize=(x_wide, y_wide), dpi=200)
    plt.pcolormesh(grid_all.T, cmap=cmap, norm=None)
    plt.axis('scaled')

    ticks_x = np.concatenate((np.arange(0, args.environment_size[1]+1, 10), np.arange(-10, args.environment_size[0]-1, -10)), 0)
    ticks_y = np.concatenate((np.arange(0, args.environment_size[3]+1, 10), np.arange(-10, args.environment_size[2]-1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)

    plt.title("Predicted States at Time %.1f s" % (i / 10.))
    plt.tight_layout()

def plot_traj(ego, mp_results, mp_methods, args, folder=None, traj_idx=None):
    """
    function for plotting the reference trajectories planned by different motion planners in the occupation grid
    """

    if traj_idx is None:
        traj_idx = len(mp_results[mp_methods[0]]["x_traj"]) - 1
    grid = ego.env.grid[:, :, [1]]
    x_traj_list = []
    for method in mp_methods:
        if mp_results[method]["x_traj"][traj_idx] is None:
            x_traj_list.append(None)
            continue
        x_traj = np.array(mp_results[method]["x_traj"][traj_idx].detach())
        x0 = x_traj[:, :, [0]]
        idx_old = np.linspace(0, x_traj.shape[2]-1, x_traj.shape[2])
        idx_new = np.linspace(0, x_traj.shape[2]-1, (x_traj.shape[2] -1) * 10 + 1)
        x_traj_long = np.zeros((1, x_traj.shape[1], (x_traj.shape[2] -1) * 10 + 1))
        for j in range(x_traj.shape[1]):
            x_traj_long[0, j, :] = np.interp(idx_new, idx_old, x_traj[0, j, :])
        x_traj_list.append(torch.from_numpy(x_traj_long))

    colorarray = np.concatenate((convert_color(TUMBlue),
                                 convert_color(TUMGray),
                                 convert_color(TUMGreen_acc),
                                 convert_color(TUMOrange_acc),
                                 convert_color(TUMBlue_light),
                                 convert_color(TUMGray_light),
                                 convert_color(TUMBlue_med)))
    col_start = convert_color(MITRed)
    grid_all = 1 - np.repeat(grid, 4, axis=2)
    grid_all[:, :, 3] = 1

    plt.close("all")
    if args.mp_use_realEnv:
        div = 6
    else:
        div = 4
    x_wide = np.abs((args.environment_size[1] - args.environment_size[0])) / div
    y_wide = np.abs((args.environment_size[3] - args.environment_size[2])) / div
    plt.figure(figsize=(x_wide, y_wide), dpi=300)
    for i, x_traj in enumerate(x_traj_list):
        if mp_methods[i] == "grad" and "search" in mp_methods:
            label = "Gradient-based Method" #"Density planner"
        elif mp_methods[i] == "search":
            label = "Search-based Method"
        elif mp_methods[i] == "sampl":
            label = "Sampling-based Method"
        elif mp_methods[i] == "grad":
            label = "Density planner"
        elif mp_methods[i] == "oracle":
            label = "Oracle"
        elif mp_methods[i] == "tube2MPC":
            label = "MPC with $r_\\textrm{tube}=0.5m$"
        elif mp_methods[i] == "tube3MPC":
            label = "MPC with $r_\\textrm{tube}=1m$"
        else:
            label = mp_methods[i]
        plt.plot(0, 0, "-", color=colorarray[i, :], label=label)
        if x_traj is None:
            continue
        grid_traj = traj2grid(x_traj, args)
        idx = grid_traj != 0
        grid_idx = grid_all[idx]
        grid_idx[:, :] = torch.from_numpy(colorarray[[i], :]) #.unsqueeze(0)
        grid_all[idx] = grid_idx

    plt.imshow(torch.transpose(grid_all, 0, 1), origin="lower")
    gridpos_x, gridpos_y = pos2gridpos(args, pos_x=[x0[0, 0, 0], ego.xrefN[0, 0, 0]],
                                       pos_y=[x0[0, 1, 0], ego.xrefN[0, 1, 0]])
    plt.scatter(gridpos_x[0], gridpos_y[0], c=col_start, marker='o', s=80, label="Start")
    plt.scatter(gridpos_x[1], gridpos_y[1], c=col_start, marker='x', s=100, label="Goal")
    plt.axis('scaled')
    if args.mp_use_realEnv == False:
        plt.legend(bbox_to_anchor=(0.5, -0.11), loc="upper center")
    elif args.mp_recording == 26:
        plt.legend(loc="upper right")
    else:
        plt.legend(loc="upper left")

    ticks_x = np.concatenate((np.arange(0, args.environment_size[1]+1, 10), np.arange(-10, args.environment_size[0]-1, -10)), 0)
    ticks_y = np.concatenate((np.arange(0, args.environment_size[3]+1, 10), np.arange(-10, args.environment_size[2]-1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)
    plt.xlabel("$p_x$")
    plt.ylabel("$p_y$")

    #plt.title(f"{name}" + str_timestep)
    plt.tight_layout()
    if folder is None:
        folder = args.path_plot_grid
    filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_GridTraj%d" % traj_idx + ".jpg"
    plt.savefig(folder + filename, dpi=200)
    plt.clf()
