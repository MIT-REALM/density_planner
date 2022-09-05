import numpy as np
import torch
from motion_planning.utils import pos2gridpos, traj2grid, shift_array, \
    pred2grid, get_mesh_sample_points, sample_pdf, enlarge_grid, compute_gradient
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from plots.plot_functions import plot_ref, plot_grid, plot_motion
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from systems.sytem_CAR import Car


class Environment:
    """Combined grip map of multiple OccupancyObjects"""

    def __init__(self, objects, args, name="environment", timestep=0):
        # self.time = time
        self.grid_size = args.grid_size
        self.current_timestep = timestep
        self.objects = objects
        self.update_grid()
        self.name = name
        self.grid_enlarged = None
        self.grid_gradientX = None
        self.grid_gradientY = None

    def update_grid(self):
        """forward object occupancies to the current timestep and add all grids together"""
        number_timesteps = self.current_timestep + 1
        self.grid = torch.zeros((self.grid_size[0], self.grid_size[1], number_timesteps))
        for obj in self.objects:
            if obj.grid.shape[2] < number_timesteps:
                obj.forward_occupancy(step_size=number_timesteps - obj.grid.shape[2])
            self.add_grid(obj.grid)

    def add_grid(self, grid):
        """add individual object grid to the overall occupancy grid"""
        self.grid = torch.clamp(self.grid + grid[:, :, :self.current_timestep + 1], 0, 1)

    def add_object(self, obj):
        """add individual object to the environment and add occupancy grid"""
        if obj.grid_size == self.grid_size and isinstance(obj, StaticObstacle):
            self.objects.append(obj)
            self.add_grid(obj.grid)
        else:
            raise Exception("Gridsize must match and object must be of type StaticObstacle!")

    def forward_occupancy(self, step_size=1):
        """increment current time step and update the grid"""
        self.current_timestep += step_size
        self.update_grid()

    def enlarge_shape(self, table=None):
        """enlarge the shape of all obstacles and update the grid to do motion planning for a point"""
        if table is None:
            table = [[0, 10, 25], [10, 30, 20], [30, 50, 10], [50, 101, 5]]
        grid_enlarged = self.grid.clone().detach()
        for elements in table:
            if elements[0] >= self.grid.shape[2]:
                continue
            timesteps = torch.arange(elements[0], min(elements[1], self.grid.shape[2]))
            grid_enlarged[:, :, timesteps] = enlarge_grid(self.grid[:, :, timesteps], elements[2])
        self.grid_enlarged = grid_enlarged

    def get_gradient(self):
        if self.grid_gradientX is None:
            grid_gradientX, grid_gradientY = compute_gradient(self.grid, step=1)
            s = 5
            missingGrad = torch.logical_and(self.grid != 0, torch.logical_and(grid_gradientX == 0, grid_gradientY == 0))
            while torch.any(missingGrad):
                idx = missingGrad.nonzero(as_tuple=True)
                grid_gradientX_new, grid_gradientY_new = compute_gradient(self.grid, step=s)
                grid_gradientX[idx] += s * grid_gradientX_new[idx]
                grid_gradientY[idx] += s * grid_gradientY_new[idx]
                s += 10
                missingGrad = torch.logical_and(self.grid != 0, torch.logical_and(grid_gradientX == 0, grid_gradientY == 0))

            self.grid_gradientX = grid_gradientX
            self.grid_gradientY = grid_gradientY

class StaticObstacle:
    """Grid map of certain object with predicted occupancy probability value for each cell and timestep"""

    def __init__(self, args, coord, name="staticObstacle", timestep=0):
        self.grid_size = args.grid_size
        self.current_timestep = 0
        self.name = name
        self.grid = torch.zeros((self.grid_size[0], self.grid_size[1], timestep + 1))
        self.bounds = [None for _ in range(timestep + 1)]
        self.occupancies = []
        self.add_occupancy(args, pos=coord[0:4], certainty=coord[4], spread=coord[5])

    def add_occupancy(self, args, pos, certainty=1, spread=1, pdf_form='square'):
        grid_pos_x, grid_pos_y = pos2gridpos(args, pos[:2], pos[2:])
        normalise = False
        if certainty is None:
            certainty = 1
            normalise = True
        if pdf_form == 'square':
            for i in range(int(spread)):
                min_x = max(grid_pos_x[0] - i, 0)
                max_x = min(grid_pos_x[1] + i + 1, self.grid.shape[0])
                min_y = max(grid_pos_y[0] - i, 0)
                max_y = min(grid_pos_y[1] + i + 1, self.grid.shape[1])
                self.grid[min_x:max_x, min_y:max_y, self.current_timestep] += certainty / spread
            limits = torch.tensor([grid_pos_x[0] - i, grid_pos_x[1] + i, grid_pos_y[0] - i, grid_pos_y[1] + i])
            if self.bounds[self.current_timestep] is None:
                self.bounds[self.current_timestep] = limits
            else:
                self.bounds[self.current_timestep] = torch.cat((self.bounds[self.current_timestep],
                                                                limits), dim=0)

                # self.nonzero_gridpos[self.current_timestep].append()
                # = grid[grid_pos_x[0] - i:grid_pos_x[1] + i, grid_pos_y[0] - i:grid_pos_y[1] + i, :] + certainty / spread
        else:
            raise (NotImplementedError)

        if normalise:
            self.grid[:, :, self.current_timestep] /= self.grid[:, :, self.current_timestep].sum()

    def forward_occupancy(self, step_size=1):
        self.grid = torch.cat((self.grid, torch.repeat_interleave(self.grid[:, :, [self.current_timestep]], step_size,
                                                                  dim=2)), dim=2)
        self.bounds += [self.bounds[self.current_timestep]] * step_size
        self.current_timestep += step_size

    def enlarge_shape(self, wide):
        pass


class DynamicObstacle(StaticObstacle):
    def __init__(self, args, coord, name="dynamicObstacle", timestep=0, velocity_x=0, velocity_y=0):
        super().__init__(args, coord, name=name, timestep=timestep)
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def forward_occupancy(self, step_size=1):
        self.grid = torch.cat((self.grid, torch.zeros((self.grid_size[0], self.grid_size[1], step_size))), dim=2)
        for i in range(step_size):
            self.grid[:, :, self.current_timestep + 1 + i] = shift_array(self.grid[:, :, self.current_timestep + i],
                                                                         self.velocity_x, self.velocity_y)
            self.bounds.append(self.bounds[self.current_timestep + i] +
                               torch.tensor([self.velocity_x, self.velocity_x, self.velocity_y, self.velocity_y]))
        self.current_timestep += step_size


class EgoVehicle:
    def __init__(self, xref0, xrefN, env, args, pdf0=None, name="egoVehicle", video=False):
        self.xref0 = xref0
        self.xrefN = xrefN
        self.system = Car()
        self.name = name
        self.env = env
        self.args = args
        self.video = video
        if pdf0 is None:
            pdf0 = sample_pdf(self.system, args.mp_gaussians)
        self.initialize_predictor(pdf0)
        self.env.get_gradient()

    def initialize_predictor(self, pdf0):
        self.model = self.load_predictor(self.system.DIM_X)
        if self.args.mp_sampling == 'random':
            xe0 = torch.rand(self.args.mp_sample_size, self.system.DIM_X, 1) * (
                    self.system.XE0_MAX - self.system.XE0_MIN) + self.system.XE0_MIN
        else:
            _, xe0 = get_mesh_sample_points(self.system, self.args)
            xe0 = xe0.unsqueeze(-1)
        rho0 = pdf0(xe0)
        mask = rho0 > 0
        self.xe0 = xe0[mask, :, :]
        self.rho0 = (rho0[mask] / rho0.sum()).reshape(-1, 1, 1)

    def load_predictor(self, dim_x):
        _, num_inputs = load_inputmap(dim_x, self.args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, self.args, load_pretrained=True)
        model.eval()
        return model

    def predict_density(self, up, xref_traj, use_nn=True, xe0=None, rho0=None, compute_density=True):
        if xe0 is None:
            xe0 = self.xe0
        if rho0 is None:
            rho0 = self.rho0
            assert rho0.shape[0] == xe0.shape[0]

        if self.args.input_type == "discr10" and up.shape[2] != 10:
            N_sim = up.shape[2] * (self.args.N_sim_max // 10) + 1
            up = torch.cat((up, torch.zeros(up.shape[0], up.shape[1], 10 - up.shape[2])), dim=2)
        elif self.args.input_type == "discr5" and up.shape[2] != 5:
            N_sim = up.shape[2] * (self.args.N_sim_max // 5) + 1
            up = torch.cat((up, torch.zeros(up.shape[0], up.shape[1], 5 - up.shape[2])), dim=2)
        else:
            N_sim = self.args.N_sim

        if use_nn: # approximate with density NN
            xe_traj = torch.zeros(xe0.shape[0], xref_traj.shape[1], xref_traj.shape[2])
            rho_log_unnorm = torch.zeros(xe0.shape[0], 1, xref_traj.shape[2])
            t_vec = torch.arange(0, self.args.dt_sim * N_sim - 0.001, self.args.dt_sim * self.args.factor_pred)

            # 2. predict x(t) and rho(t) for times t
            for idx, t in enumerate(t_vec):
                xe_traj[:,:, [idx]], rho_log_unnorm[:, :, [idx]] = get_nn_prediction(self.model, xe0[:, :, 0], self.xref0[0, :, 0], t, up, self.args)
        else: # use LE
            uref_traj, _ = self.system.sample_uref_traj(self.args, up=up)
            xref_traj_long = self.system.compute_xref_traj(self.xref0, uref_traj, self.args)
            xe_traj_long, rho_log_unnorm = self.system.compute_density(xe0, xref_traj_long, uref_traj, self.args.dt_sim,
                                                   cutting=False, log_density=True, compute_density=True)
            xe_traj = xe_traj_long[:, :, ::self.args.factor_pred]

        if compute_density:
            rho_max, _ = rho_log_unnorm.max(dim=0)
            rho_unnorm = torch.exp(rho_log_unnorm - rho_max.unsqueeze(0)) * rho0.reshape(-1, 1, 1)
            rho_traj = rho_unnorm / rho_unnorm.sum(dim=0).unsqueeze(0)
            rho_traj = rho_traj #+ rho0.reshape(-1, 1, 1)
        else:
            rho_traj = None

        x_traj = xe_traj + xref_traj
        return x_traj, rho_traj

    def visualize_xref(self, xref_traj, uref_traj=None, show=True, save=False, include_date=True,
                       name='Reference Trajectory', folder=None):
        if uref_traj is not None:
            plot_ref(xref_traj, uref_traj, 'Reference Trajectory', self.args, self.system, t=self.t_vec,
                     include_date=True)
        grid = traj2grid(xref_traj, self.args)
        grid_env_max, _ = self.env.grid.max(dim=2)
        plot_grid(torch.clamp(grid + grid_env_max, 0, 1), self.args, name=name,
                  show=show, save=save, include_date=include_date, folder=folder)

    def animate_traj(self, folder, xref_traj, x_traj=None, rho_traj=None):
        if x_traj is None:
            x_traj = xref_traj
        if rho_traj is None:
            rho_traj = torch.ones(x_traj.shape[0], 1, x_traj.shape[2]) / x_traj.shape[0]  # assume equal density

        # create colormap
        greys = cm.get_cmap('Greys')
        grey_col = greys(range(0, 256))
        greens = cm.get_cmap('Greens')
        green_col = greens(range(0, 256))
        blue = np.array([[0.212395, 0.359683, 0.55171, 1.]])
        # yellow = np.array([[0.993248, 0.906157, 0.143936, 1.      ]])
        colorarray = np.concatenate((grey_col[::2, :], green_col[::2, :], blue))
        cmap = ListedColormap(colorarray)

        grid_env_sc = 127 * self.env.grid

        if self.video:

            animation_1 = animation.FuncAnimation(plt.gcf(), plot_motion, fargs=(cmap, x_traj, rho_traj, xref_traj, self.args, grid_env_sc), frames=xref_traj.shape[2], interval=2000./xref_traj.shape[2])
            #animation_1.save('output2.gif', writer='imagemagick')
            animation_1.save('output1.mp4', writer='ffmpeg')
            plt.show()

        for i in range(xref_traj.shape[2]):
            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_pred = pred2grid(x_traj[:, :, [i]], rho_traj[:, :, [i]], self.args, return_gridpos=False)

            grid_pred_sc = 127 * torch.clamp(grid_pred/grid_pred.max(), 0, 1)
            grid_pred_sc[grid_pred_sc != 0] += 128
            grid_traj = traj2grid(xref_traj[:, :, :i + 1], self.args)
            grid_traj[grid_traj != 0] = 256
            grid_all = torch.clamp(grid_env_sc[:, :, i] + grid_traj + grid_pred_sc[:, :, 0], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)

    def set_start_grid(self):  # certainty=None, spread=2, pdf_form='squared'):
        self.grid = pred2grid(self.xref0 + self.xe0, self.rho0, self.args)  # 20s for 100iter
        # self.grid = pdf2grid(pdf0, self.xref0, self.system, args) #65s for 100iter

