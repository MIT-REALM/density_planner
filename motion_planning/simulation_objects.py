import numpy as np
import torch
from motion_planning.utils import pos2gridpos, check_collision, idx2time, gridpos2pos, traj2grid, shift_array, \
    pred2grid, get_mesh_sample_points, time2idx, sample_pdf, enlarge_grid, get_closest_free_cell, get_closest_obs_cell
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from systems.utils import listDict2dictList
from plots.plot_functions import plot_ref, plot_grid, plot_cost
import pickle
from datetime import datetime
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import shutil
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
            limits = torch.tensor([[min_x, max_x - 1, min_y, max_y - 1]])
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
            self.bounds.append(self.bounds[self.current_timestep] +
                               torch.tensor([[self.velocity_x, self.velocity_x, self.velocity_y, self.velocity_y]]))
        self.current_timestep += step_size


class MPOptimizer:
    def __init__(self, ego):
        self.ego = ego
        self.found_traj = []
        if ego.args.input_type == "discr10":
            self.num_repeats = 100 # number of timesteps one input signal is hold
        elif ego.args.input_type == "discr5":
            self.num_repeats = 200
        self.num_discr = self.ego.args.N_sim // self.num_repeats  # number of discretizations for the whole prediction period
        self.u0 = torch.arange(self.ego.system.UREF_MIN[0, 0, 0] + 1, self.ego.system.UREF_MAX[0, 0, 0] - 1 + 1e-5,
                               self.ego.args.du_search[0])
        self.u1 = torch.arange(self.ego.system.UREF_MIN[0, 1, 0] + 1, self.ego.system.UREF_MAX[0, 1, 0] - 1 + 1e-5,
                               self.ego.args.du_search[1])

    def search_good_traj(self, num_traj=None, plot=False):
        i = 0
        self.num_traj = num_traj
        self.plot = plot

        self.ego.path_log_search = self.ego.path_log + "_search" + "/"
        os.makedirs(self.ego.path_log_search)

        for u0 in self.u0:
            for u1 in self.u1:
                u_ext = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.num_repeats)
                self.check_uref(u_ext, self.ego.xref0, i)
                if len(self.found_traj) == self.num_traj:
                    return
        with open(self.ego.path_log_search + "found_traj", "wb") as f:
            pickle.dump(self.found_traj, f)

    def find_best_traj(self, load=True, valid_traj=None):
        if load:
            with open(self.ego.path_log_search + "found_traj", "rb") as f:
                valid_traj = pickle.load(f)
        else:
            if valid_traj is None:
                if len(self.found_traj) == 0:
                    self.search_good_traj()
                valid_traj = self.found_traj

        cost_min = np.inf
        for i, traj in enumerate(valid_traj):
            if traj[1].item() < cost_min:
                cost_min = traj[1].item()
                idx_min = i
        u_params = valid_traj[idx_min][0]
        with open(self.ego.path_log_search + "best_traj", "wb") as f:
            pickle.dump([u_params, cost_min], f)
        return u_params, cost_min

    def plot_found_traj(self):
        foldername = "foundTraj" + "/"
        folder = os.path.join(self.ego.path_log_search, foldername)
        os.makedirs(folder)
        for item in self.found_traj:
            u_params = item[0]
            cost = item[1]
            uref_traj, xref_traj = self.system.u_params2ref_traj(self.ego.xref0, u_params, self.ego.args, short=True)
            self.ego.visualize_xref(xref_traj, name="cost%.2f" % cost, save=True, show=False, folder=folder)


    def check_uref(self, uref_traj, xref_traj, i):
        distance_to_goal_old = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()

        # 1. compute xref
        xref_traj = self.ego.system.extend_xref_traj(xref_traj, uref_traj, self.ego.args.dt_sim)
        if xref_traj is None:  # = out of admissible state space
            return "invalid"

        # 2. check if trajectory is promising
        distance_to_goal = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()
        if distance_to_goal > distance_to_goal_old:
            return "invalid"
        heading_to_goal = np.arctan2(self.ego.xrefN[0, 1, 0] - xref_traj[0, 1, -1],
                                     self.ego.xrefN[0, 0, 0] - xref_traj[0, 0, -1])
        heading_diff = (xref_traj[0, 2, -1] - heading_to_goal + np.pi) % (2 * np.pi) - np.pi
        if heading_diff.abs() > np.pi / 2 + 0.3:
            return "invalid"
        collision, _, _ = self.ego.find_collision_xref(xref_traj, i * self.num_repeats, (i+1) * self.num_repeats)
        if collision:
            return "invalid"

        i += 1
        if i == self.num_discr:
            cost = distance_to_goal + 0.01 * (uref_traj ** 2).sum()
            if float(distance_to_goal) < self.ego.args.cost_threshold:
                u_params = uref_traj[:, :, ::self.num_repeats]
                if len(self.found_traj) > 0 and (self.found_traj[-1][0][0, 0, :] - u_params[0, 0, :]).abs().sum() < 3:
                    #(self.found_traj[-1][0] - u_params).abs().sum() < 4:
                    # try:
                    # - only consider yaw rate for the difference:
                    # - certain number of trajectories which lead to collision between two different traj.
                    if self.found_traj[-1][1] > cost:
                        self.found_traj[-1] = [u_params, cost]
                        str = "updatedTraj_cost%.2f" % cost
                else:
                    self.found_traj.append([u_params, cost])
                    str = "addedTraj_cost%.2f" % cost
            print(str)
            if self.plot:
                self.ego.visualize_xref(xref_traj, show=False, save=True, name=str, folder=self.ego.path_log_search)
            return

        # 4. extend traj with u0 and u1
        for u0 in self.u0:
            for u1 in self.u1:
                u_ext = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.num_repeats)
                uref_traj_new = torch.cat((uref_traj, u_ext), dim=2)

                # 5. call check_uref
                self.check_uref(uref_traj_new, xref_traj, i)
                if len(self.found_traj) == self.num_traj:
                    return

    def optimize_traj(self, u_params, with_density=True, uparams_name=""):
        u_params.requires_grad = True
        costs = []
        costs_dict = []
        self.coll_pos = []
        self.other_costs = False

        if with_density:
            name = "_optDensity"
        else:
            name = "_opt"
        self.ego.path_log_opt = self.ego.path_log + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + name + uparams_name + "/"
        os.makedirs(self.ego.path_log_opt)
        with open(self.ego.path_log_opt + "initial_uparams", "wb") as f:
            pickle.dump(u_params, f)

        for i in range(self.ego.args.epochs_mp):
            cost, cost_dict = self.predict_cost(u_params, iter=i, with_density=with_density)
            if i > 10 and self.other_costs and cost_dict["cost_coll"] / self.ego.args.cost_coll_weight < 1e-5 and \
                    cost_dict["cost_goal"] / self.ego.args.cost_goal_weight < 4:
                break
            cost.backward()
            costs.append(cost.item())
            costs_dict.append(cost_dict)
            with torch.no_grad():
                lr = self.ego.args.lr_mp * (20 / (i + 20))
                u_params -= torch.clamp(u_params.grad * lr,
                                        -self.ego.args.max_gradient, self.ego.args.max_gradient)
                u_params.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
                u_params.grad.zero_()
        costs_dict = listDict2dictList(costs_dict)
        plot_cost(costs_dict, self.ego.args, folder=self.ego.path_log_opt)
        with open(self.ego.path_log_opt + "results", "wb") as f:
            pickle.dump([u_params, costs_dict], f)
        return u_params

    def predict_cost(self, u_params, iter=0, with_density=True):
        uref_traj, xref_traj = self.ego.system.u_params2ref_traj(self.ego.xref0, u_params, self.ego.args, short=True)
        #self.ego.animate_traj(self.ego.env.grid_enlarged, xref_traj, self.ego.args, name='test')
        name = "iter%d, lr=%.0e, \n w_goal=%.0e, w_u=%.0e\n w_coll=%.0e, w_bounds=%.0e" % (iter, self.ego.args.lr_mp, self.ego.args.cost_goal_weight,
            self.ego.args.cost_uref_weight, self.ego.args.cost_coll_weight, self.ego.args.cost_bounds_weight)
        self.ego.visualize_xref(xref_traj, name="iter%d" % iter, save=True, show=False, folder=self.ego.path_log_opt)

        # penalize leaving the admissible state space
        cost_bounds, out_of_bounds = self.get_cost_bounds(xref_traj)

        # penalize big goal distance
        cost_goal = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()
        if cost_goal < 2:
            self.other_costs = True

        # penalize big control effort
        cost_uref = (uref_traj ** 2).sum()

        cost_coll = torch.Tensor([0])
        cost_obsDist = torch.Tensor([0])

        if self.other_costs:
            # penalize collisions
            cost_coll = self.get_cost_coll(xref_traj, u_params, with_density=with_density and not out_of_bounds)

            # penalize small distances to obstacles
            if cost_coll < 1e-5:
                cost_obsDist = self.get_cost_obsDist(xref_traj)

        cost = self.ego.args.cost_goal_weight * cost_goal \
               + self.ego.args.cost_uref_weight * cost_uref \
               + self.ego.args.cost_bounds_weight * cost_bounds \
               + (max(iter, 200)/40 + 1) * self.ego.args.cost_coll_weight * cost_coll \
               + (max(iter, 200)/40 + 1) * self.ego.args.cost_obsDist_weight * cost_obsDist

        cost_dict = {
            "cost_sum": cost.item(),
            "cost_goal": self.ego.args.cost_goal_weight * cost_goal.item(),
            "cost_uref": self.ego.args.cost_uref_weight * cost_uref.item(),
            "cost_bounds": self.ego.args.cost_bounds_weight * cost_bounds.item(),
            "cost_obsDist": self.ego.args.cost_obsDist_weight * cost_obsDist.item(),
            "cost_coll": (iter / 10 + 1) * self.ego.args.cost_coll_weight * cost_coll.item()
            }
        return cost, cost_dict

    def get_cost_bounds(self, xref_traj):
        # penalize leaving the admissible state space
        cost = torch.Tensor([0])
        out_of_bounds = False
        if torch.any(xref_traj < self.ego.system.X_MIN_MP):
            idx_smaller = (xref_traj < self.ego.system.X_MIN_MP).nonzero(as_tuple=True)
            cost += ((xref_traj[idx_smaller] - self.ego.system.X_MIN_MP[0, idx_smaller[1], 0]) ** 2).sum()
            out_of_bounds = True
        if torch.any(xref_traj > self.ego.system.X_MAX_MP):
            idx_bigger = (xref_traj > self.ego.system.X_MAX_MP).nonzero(as_tuple=True)
            cost += ((xref_traj[idx_bigger] - self.ego.system.X_MAX_MP[0, idx_bigger[1], 0]) ** 2).sum()
            out_of_bounds = True
        return cost, out_of_bounds

    def get_cost_coll(self, xref_traj, u_params=None, with_density=True):
        # penalize collisions

        # 1. check if xref_traj collides with enlarged obstacles
        enlarged = with_density or self.ego.args.mp_enlarged
        collision, coll_times_idx, coll_pos_prob = self.ego.find_collision_xref(xref_traj, 0, xref_traj.shape[2],
                                                                                 enlarged=enlarged, short=True,
                                                                                 check_all=with_density,
                                                                                 return_coll_pos=not with_density)
        cost_coll = torch.Tensor([0])
        if collision:
            if with_density:
                # 2. check if predicted sample distribution collides with normal obstacles
                coll_times = idx2time(coll_times_idx, self.ego.args)
                collision, coll_times_idx, coll_pos_prob, pred_nn = self.ego.find_collision_xnn(u_params, xref_traj,
                                                                                                     t_vec=coll_times)
                if not collision:
                    return cost_coll
                x_nn, rho_nn, gridpos_nn = pred_nn

            # 3. compute distance from collision position to free space
            for i in range(coll_pos_prob.shape[1]):
                if with_density:
                    # compute sample indizes which collide with obstacle
                    coll_idx = (gridpos_nn == coll_pos_prob[:2, i]).all(dim=1).nonzero(as_tuple=True)[0]
                    x_coll = x_nn[coll_idx, :, :]
                    rho_coll = rho_nn[coll_idx, 0, 0]
                else:
                    x_coll = xref_traj[:, :, [coll_times_idx]]
                    rho_coll = torch.ones(1)
                if enlarged:
                    grid = self.ego.env.grid_enlarged[:, :, coll_times_idx]
                else:
                    grid = self.ego.env.grid[:, :, coll_times_idx]
                distance_to_free_cell = self.compute_dist_free_cell(x_coll, coll_pos_prob[:2, i], grid)

                #4. penalize big distances
                cost_coll += coll_pos_prob[2, i] * (rho_coll * distance_to_free_cell).sum()
        return cost_coll

    def compute_dist_free_cell(self, x, coll_pos, grid_env):
        # get position of closest free space for collision cell i
        gridpos_free = get_closest_free_cell(coll_pos, grid_env, self.ego.args)
        pos_x, pos_y = gridpos2pos(self.ego.args, gridpos_free[0], gridpos_free[1])

        distance_to_free_cell = (x[:, 0, 0] - pos_x) ** 2 + (x[:, 1, 0] - pos_y) ** 2
        return distance_to_free_cell

    def get_cost_obsDist(self, xref_traj):
        cost_obsDist = torch.Tensor([0])
        for i in range(0, xref_traj.shape[2], 5):
            gridpos_x, gridpos_y = pos2gridpos(self.ego.args, xref_traj[0, 0, i], xref_traj[0, 1, i])
            gridpos_obs = get_closest_obs_cell(torch.Tensor([gridpos_x, gridpos_y]), self.ego.env.grid_enlarged[:, :, i], self.ego.args)
            if gridpos_obs is not None:
                pos_x, pos_y = gridpos2pos(self.ego.args, gridpos_obs[0], gridpos_obs[1])
                cost_obsDist += 1 / ((xref_traj[0, 0, i] - pos_x) ** 2 + (xref_traj[0, 1, i] - pos_y) ** 2)
        return cost_obsDist


    # def compute_cost_to_coll(self, xref_traj):
    #     prob = 0
    #     for i in range(self.ego.args.addit_enlargement):
    #         collision, _, coll_pos_prob = self.ego.find_collision_xref(xref_traj, 0, xref_traj.shape[2],
    #                                         enlarged=True, short=True, check_all=True, return_coll_pos=True, add=i)
    #         if collision:
    #             for item in coll_pos_prob:
    #                 prob += 1 / (i + 1) ** 2 * item[2, :].sum()
    #     return prob


class EgoVehicle:
    def __init__(self, xref0, xrefN, env, args, pdf0=None, name="egoVehicle"):
        self.xref0 = xref0
        self.xrefN = xrefN
        self.t_vec = torch.arange(0, args.dt_sim * args.N_sim - 0.001, args.dt_sim * args.factor_pred)
        self.system = Car()
        self.name = name
        self.env = env
        self.args = args
        foldername = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_mp_" + args.mp_name + "/"
        self.path_log = os.path.join(args.path_plot_motion, foldername)
        os.makedirs(self.path_log)
        shutil.copyfile('hyperparams.py', self.path_log + 'hyperparams.py')
        shutil.copyfile('motion_planning/simulation_objects.py', self.path_log + 'simulation_objects.py')
        shutil.copyfile('motion_planning/plan_motion.py', self.path_log + 'plan_motion.py')
        self.path_log_search = None
        self.path_log_opt = None
        self.path_log_mp = None
        if pdf0 is None:
            pdf0 = sample_pdf(self.system, 5)
        self.initialize_predictor(pdf0)

    def visualize_xref(self, xref_traj, uref_traj=None, show=True, save=False, include_date=True,
                       name='Reference Trajectory', folder=None):
        if uref_traj is not None:
            plot_ref(xref_traj, uref_traj, 'Reference Trajectory', self.args, self.system, t=self.t_vec,
                     include_date=True)
        grid = traj2grid(xref_traj, self.args)
        grid_env_max, _ = self.env.grid.max(dim=2)
        plot_grid(torch.clamp(grid + grid_env_max, 0, 1), self.args, name=name,
                  show=show, save=save, include_date=include_date, folder=folder)

    def animate_traj(self, u_params, name="", folder=None):
        _, xref_traj = self.system.u_params2ref_traj(self.xref0, u_params, self.args, short=True)

        if folder is None:
            folder = self.path_log
        foldername = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_animation_" + name + "/"
        folder = os.path.join(folder, foldername)
        os.makedirs(folder)
        with open(folder + "u_params", "wb") as f:
            pickle.dump(u_params, f)

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
        for i in range(xref_traj.shape[2]):

            t = idx2time(i, self.args)
            xe_nn, rho_nn_fac = get_nn_prediction(self.model, self.xe0[:, :, 0], self.xref0[0, :, 0], t, u_params, self.args)
            x_nn = xe_nn + xref_traj[:, :, [i]]
            rho_nn = rho_nn_fac * self.rho0.reshape(-1, 1, 1)
            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_pred = pred2grid(x_nn, rho_nn, self.args, return_gridpos=False)

            grid_pred_sc = 127 * torch.clamp(grid_pred/grid_pred.max(), 0, 1)
            grid_pred_sc[grid_pred_sc != 0] += 128
            grid_traj = traj2grid(xref_traj[:, :, :i + 1], self.args)
            grid_traj[grid_traj != 0] = 256
            grid_all = torch.clamp(grid_env_sc[:, :, i] + grid_traj + grid_pred_sc[:, :, 0], 0, 256)
            plot_grid(grid_all, self.args, name="iter%d" % i, cmap=cmap, show=False, save=True, folder=folder)


    def set_start_grid(self):  # certainty=None, spread=2, pdf_form='squared'):
        self.grid = pred2grid(self.xref0 + self.xe0, self.rho0, self.args)  # 20s for 100iter
        # self.grid = pdf2grid(pdf0, self.xref0, self.system, args) #65s for 100iter

    def initialize_predictor(self, pdf0):
        self.model = self.load_predictor(self.system.DIM_X)
        if self.args.sampling == 'random':
            xe0 = torch.rand(self.sample_size, self.system.DIM_X, 1) * (
                    self.system.XE0_MAX - self.system.XE0_MIN) + self.system.XE0_MIN
        else:
            _, xe0 = get_mesh_sample_points(self.system, self.args)
            xe0 = xe0.unsqueeze(-1)
        rho0 = pdf0(xe0)
        mask = rho0 > 0
        self.xe0 = xe0[mask, :, :]
        self.rho0 = (rho0[mask] / rho0.sum()).reshape(-1, 1, 1)

    def find_collision_xref(self, xref_traj, i_start_x, i_end_x=None, enlarged=False, short=False, check_all=False,
                            return_coll_pos=False, add=0):
        """
        Checks if the reference trajectory xref is colliding with any obstacles in a specified time interval

        Parameters
        ----------
        xref_traj:      reference trajectory
            torch.Tensor: 1 x self.DIM_X x prediction_length
        i_start_x:      first trajectory time index which should be checked for collision
            int
        i_end_x:        last trajectory time index which should be checked for collision
            int
        enlarged:       true if obstacles in the environemnt should be enlarged before collision checking
            bool
        short:          true if reference trajectory is adapted to the prediction timesteps
            bool
        check_all:      false if collision checking is aborted after a collision is found
            bool
        return_coll_pos:    true if the collision position and probability is computed
            bool
        add:            number of grid cells which the obstacles get additionally enlarged
            int

        Returns
        -------
        coll:               true if collision is detected in the intervall
            bool
        coll_times_idx:     time indizes where collision is detected
            list of tensors (if check_all) or tensor
        coll_pos_probs:     tensors which contain the collision gridposition and the corresponding  collision probabilities
            torch.Tensor: 3 [x-y-prob] x number of grid cells where collision is detected
            or list of torch.Tensor (if check_all) with length=number of timesteps with collision
        """

        if enlarged:
            if self.env.grid_enlarged is None:
                self.env.enlarge_shape(add=add)
            grid = self.env.grid_enlarged
        else:
            grid = self.env.grid

        if short:
            if i_end_x is not None:
                i_end_short = i_end_x
            i_start_short = i_start_x
            interv_x = torch.Tensor([0]).long()
        else:
            if i_end_x is not None:
                i_end_short = i_end_x // self.args.factor_pred
            i_start_short = i_start_x // self.args.factor_pred
            interv_x = torch.arange(0, self.args.factor_pred)

        if i_end_x is None:
            i_end_short = i_start_short + 1
        length_interv_x = len(interv_x)
        coll_pos_probs = []
        coll_times_idx = torch.Tensor([])
        coll = False
        for i_short in range(i_start_short, i_end_short):
            interv_i = i_short * length_interv_x + interv_x
            grid_xref = traj2grid(xref_traj[:, :, i_short * length_interv_x + interv_x], self.args)
            collision, _, coll_pos_prob = check_collision(grid_xref, grid[:, :, i_short], self.args.coll_sum,
                                                          return_coll_pos=return_coll_pos)

            if collision:
                coll = True
                if not check_all:
                    return coll, interv_i, coll_pos_prob
                if return_coll_pos:
                    coll_pos_probs.append(coll_pos_prob)
                coll_times_idx = torch.cat((coll_times_idx, interv_i))

        return coll, coll_times_idx, coll_pos_probs

    def find_collision_xnn(self, u_params, xref_traj, t_vec=None):
        """
        Checks if the predicted sample distribution is colliding with any obstacles for the specified times

        Parameters
        ----------
        u_params:       parameters to represent the reference trajectory
            torch.Tensor: 1 x DIM_U x num_discr
        xref_traj:      reference trajectory
            torch.Tensor: 1 x DIM_X x prediction_length
        t_vec:          times which should be checked for collision
            torch.Tensor

        Returns
        -------
        coll:           true if collision is detected
            bool
        idx:            time index where collision is first detected
            int
        coll_pos_probs: tensors which contain the collision gridpositions and the corresponding  collision probabilities
            torch.Tensor: 3 [x-y-prob] x number of grid cells where collision is detected
            or list of torch.Tensor (if check_all) with length=number of timesteps with collision
        """

        if t_vec is None:
            t_vec = self.t_vec

        # 2. predict x(t) and rho(t) for times t
        for t in t_vec:
            idx = time2idx(t, self.args)
            xe_nn, rho_nn_fac = get_nn_prediction(self.model, self.xe0, self.xref0, t, u_params, self.args)
            x_nn = xe_nn + xref_traj[:, :, [idx]]
            rho_nn = rho_nn_fac * self.rho0.reshape(-1, 1, 1)

            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_nn, gridpos_nn = pred2grid(x_nn[:, :, [0]], rho_nn, self.args, return_gridpos=True)

                # 4. test for collisions
                collision, _, coll_pos_prob = check_collision(grid_nn[:, :, 0], self.env.grid[:, :, idx],
                                                              self.args.coll_sum, return_coll_pos=True)

            # compute collision cost penalizing the distance from samples to free space
            if collision:
                return collision, idx, coll_pos_prob, [x_nn, rho_nn, gridpos_nn]
        return 0, None, None, None

    def load_predictor(self, dim_x):
        _, num_inputs = load_inputmap(dim_x, self.args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, self.args, load_pretrained=True)
        model.eval()
        return model
