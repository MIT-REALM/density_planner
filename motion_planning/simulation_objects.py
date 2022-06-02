import numpy as np
import torch
from motion_planning.utils import pos2gridpos, check_collision, idx2time, gridpos2pos, traj2grid, shift_array,\
    pred2grid, get_mesh_sample_points, time2idx, sample_pdf, enlarge_grid, get_closest_free_cell
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from plots.plot_functions import plot_ref
from plots.plot_functions import plot_grid
import pickle


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
            limits = torch.tensor([[min_x, max_x-1, min_y, max_y-1]])
            if self.bounds[self.current_timestep] is None:
                self.bounds[self.current_timestep] = limits
            else:
                self.bounds[self.current_timestep] = torch.cat((self.bounds[self.current_timestep],
                                                                limits), dim=0)

                #self.nonzero_gridpos[self.current_timestep].append()
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
            self.grid[:, :, self.current_timestep + 1 + i] = shift_array(self.grid[:, :, self.current_timestep + i], self.velocity_x, self.velocity_y)
            self.bounds.append(self.bounds[self.current_timestep] +
                               torch.tensor([[self.velocity_x, self.velocity_x, self.velocity_y, self.velocity_y]]))
        self.current_timestep += step_size


class MPOptimizer:
    def __init__(self, ego, args):
        self.ego = ego
        self.valid_traj = []
        self.args = args
        self.num_repeats = 100  # number of timesteps one input signal is hold
        self.num_discr = args.N_sim // self.num_repeats  # number of discretizations for the whole prediction period
        self.u0 = torch.arange(self.ego.system.UREF_MIN[0, 0, 0] + 1, self.ego.system.UREF_MAX[0, 0, 0] - 1 + 1e-5,
                                    args.du_search[0])
        self.u1 = torch.arange(self.ego.system.UREF_MIN[0, 1, 0] + 1, self.ego.system.UREF_MAX[0, 1, 0] - 1 + 1e-5,
                                    args.du_search[1])

    def search_good_traj(self):
        i = 0
        for u0 in self.u0:
            for u1 in self.u1:
                u_ext = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.num_repeats)
                self.check_uref(u_ext, self.ego.xref0, i)
        with open(self.args.path_traj0 + "valid_traj", "wb") as f:
            pickle.dump(self.valid_traj, f)

    def check_uref(self, uref_traj, xref_traj, i):
        distance_to_goal_old = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()

        # 1. compute xref
        xref_traj = self.ego.system.extend_xref_traj(xref_traj, uref_traj, self.args.dt_sim)
        if xref_traj is None:  # = out of admissible state space
            return "invalid"

        # 2. check if trajectory is promising
        distance_to_goal = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()
        if distance_to_goal > distance_to_goal_old:
            return "invalid"
        heading_to_goal = np.arctan2(self.ego.xrefN[0, 1, 0] - xref_traj[0, 1, -1] , self.ego.xrefN[0, 0, 0] - xref_traj[0, 0, -1])
        heading_diff = (xref_traj[0, 2, -1] - heading_to_goal + np.pi) % (2 * np.pi) - np.pi
        if heading_diff.abs() > np.pi/2:
            return "invalid"
        collision = self.check_collision_in_interv(xref_traj, i)
        if collision:
            return "invalid"

        i += 1
        if i == self.num_discr:
            cost = distance_to_goal + 0.01 * (uref_traj ** 2).sum()
            if float(distance_to_goal) < 60:
                self.valid_traj.append([uref_traj, xref_traj, cost])
                print("valid trajectory added with cost %.2f" % cost)
                self.ego.visualize_xref(xref_traj, self.args, show=False, save=True, include_date=True, name="initialTraj")
            return

        # 4. extend traj with u0 and u1
        for u0 in self.u0:
            for u1 in self.u1:
                u_ext = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.num_repeats)
                uref_traj_new = torch.cat((uref_traj, u_ext), dim=2)

                # 5. call check_uref
                self.check_uref(uref_traj_new, xref_traj, i)


    def check_collision_in_interv(self, xref_traj, i_start, i_end=None, enlarged=False, short=False, check_all=False, return_coll_pos=False):
        if i_end is None:
            i_end = i_start + 1
        if np.isinf(i_end) or i_end > self.num_discr:
            i_end = self.num_discr
        if enlarged:
            if self.ego.env.grid_enlarged is None:
                self.ego.env.enlarge_shape()
            grid = self.ego.env.grid_enlarged
        else:
            grid = self.ego.env.grid
        num_repeats = self.num_repeats // self.args.factor_pred if short else self.num_repeats

        coll = False
        coll_time = torch.tensor([])
        for i in range(i_start, i_end):
            timestep = i * num_repeats
            for j in range(num_repeats // self.args.factor_pred):
                interv = torch.arange(timestep, timestep + self.args.factor_pred)
                grid_xref = traj2grid(xref_traj[:, :, interv], self.args)
                collision, _ = check_collision(grid_xref, grid[:, :, timestep // self.args.factor_pred], self.args.coll_sum, return_coll_pos=False)
                if collision:
                    if not check_all:
                        return True
                    coll = True
                    coll_time = torch.cat((coll_time, interv))
                timestep += self.args.factor_pred
        if not check_all:
            return False
        return coll, coll_time

    def optimize_traj(self, u_params, with_density=True):
        u_params.requires_grad = True
        costs = []
        for i in range(self.args.epochs_mp):
            cost = self.predict_cost(u_params, with_density=with_density)
            cost.backward()
            costs.append(cost.item())
            with torch.no_grad():
                u_params -= u_params.grad * self.args.lr_mp
                u_params.clamp(self.ego.system.UREF_MIN, self.ego.system.UREF_MAX)
                u_params.grad.zero_()

    def predict_cost(self, u_params, with_density=True):
        uref_traj, xref_traj = self.ego.system.u_params2ref_traj(self.ego.xref0, u_params, self.args, short=True)
        self.ego.visualize_xref(xref_traj, self.args, name="lr=%.0e, coll_weight=%.0e, \nbounds_weight=%.0e" %
                                        (self.args.lr_mp, self.args.cost_coll_weight, self.args.cost_bounds_weight))
        cost = 0

        # penalize leaving the admissible state space
        out_of_bounds = False
        if torch.any(xref_traj < self.ego.system.X_MIN):
            idx_smaller = (xref_traj < self.ego.system.X_MIN).nonzero(as_tuple=True)
            cost += self.args.cost_bounds_weight * ((xref_traj[idx_smaller] - self.ego.system.X_MIN[0, idx_smaller[1], 0]) ** 2).sum()
            out_of_bounds = True
        if torch.any(xref_traj > self.ego.system.X_MAX):
            idx_bigger = (xref_traj > self.ego.system.X_MAX).nonzero(as_tuple=True)
            cost += self.args.cost_bounds_weight * ((xref_traj[idx_bigger] - self.ego.system.X_MAX[0, idx_bigger[1], 0]) ** 2).sum()
            out_of_bounds = True

        # penalize big goal distance and big control effort
        distance_to_goal = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()
        cost += distance_to_goal + self.args.cost_uref_weight * (uref_traj ** 2).sum()
        if out_of_bounds:
            return cost

        # penalize collisions
        coll_enlarged, coll_idx = self.check_collision_in_interv(xref_traj, 0, np.inf, enlarged=True, short=True, check_all=True)  # check collision for xref


        if coll_enlarged:
            if with_density:
                coll_times = idx2time(coll_idx, self.args)
                results, idx, coll_cost = self.ego.predict_collision(u_params, xref_traj, self.args, t_vec=coll_times)
            else:
                gridpos_free = get_closest_free_cell(coll_pos_prob.T[i, :2], self.env.grid[:, :, idx], args)

            if results != 'success':
                cost += self.args.cost_coll_weight * coll_cost
        return cost


class EgoVehicle:
    def __init__(self, xref0, xrefN, pdf0, t_vec, system, env, args, name="egoVehicle"):
        self.xref0 = xref0
        self.xrefN = xrefN
        self.t_vec = t_vec
        self.system = system
        self.name = name
        self.env = env
        self.initialize_predictor(pdf0, args)


    def visualize_xref(self, xref_traj, args, uref_traj=None, show=True, save=False, include_date=True, name='Reference Trajectory'):
        if uref_traj is not None:
            plot_ref(xref_traj, uref_traj, 'Reference Trajectory', args, self.system, t=self.t_vec, include_date=True)
        grid = traj2grid(xref_traj, args)
        grid_env_max, _ = self.env.grid.max(dim=2)
        plot_grid(torch.clamp(grid + grid_env_max, 0, 1),  args, name=name,
                  show=show, save=save, include_date=include_date)

    def set_start_grid(self, args):  # certainty=None, spread=2, pdf_form='squared'):
        self.grid = pred2grid(self.xref0 + self.xe0, self.rho0, args) #20s for 100iter
        #self.grid = pdf2grid(pdf0, self.xref0, self.system, args) #65s for 100iter

    def initialize_predictor(self, pdf0, args):
        self.model = self.load_predictor(args, self.system.DIM_X)
        if args.sampling == 'random':
            xe0 = torch.rand(self.sample_size, self.system.DIM_X, 1) * (
                        self.system.XE0_MAX - self.system.XE0_MIN) + self.system.XE0_MIN
        else:
            _, xe0 = get_mesh_sample_points(self.system, args)
            xe0 = xe0.unsqueeze(-1)
        rho0 = pdf0(xe0)
        mask = rho0 > 0
        self.xe0 = xe0[mask, :, :]
        self.rho0 = (rho0[mask] / rho0.sum()).reshape(-1, 1, 1)

    def predict_collision(self, u_params, xref_traj, args, t_vec=None):
        if t_vec is None:
            t_vec = self.t_vec
        # 2. predict x(t) and rho(t) for times t
        for t in t_vec:
            idx = time2idx(t, args)
            xe_nn, rho_nn_fac = get_nn_prediction(self.model, self.xe0, self.xref0, t, u_params, args)
            x_nn = xe_nn + xref_traj[:, :, [idx]]
            rho_nn = rho_nn_fac * self.rho0.reshape(-1, 1, 1)

            with torch.no_grad():
                # 3. compute marginalized density grid
                grid_nn, gridpos_nn = pred2grid(x_nn[:, :, [0]], rho_nn, args, return_gridpos=True)

                # 4. test for collisions
                collision, coll_sum, coll_pos_prob = check_collision(grid_nn[:, :, 0], self.env.grid[:, :, idx], args.coll_sum, return_coll_pos=True)

            # compute collision cost penalizing the distance from samples to free space
            if collision:
                coll_cost = 0
                for i in range(coll_pos_prob.shape[1]):

                    # get position of closest free space for collision cell i
                    gridpos_free = get_closest_free_cell(coll_pos_prob.T[i, :2], self.env.grid[:, :, idx], args)
                    pos_x, pos_y = gridpos2pos(args, gridpos_free[0], gridpos_free[1])

                    # compute sample indizes which collide with obstacle
                    coll_idx = (gridpos_nn == coll_pos_prob.T[i, :2]).all(dim=1).nonzero(as_tuple=True)[0]
                    distance_to_free_cell = (x_nn[coll_idx, 0, 0] - pos_x) ** 2 + (x_nn[coll_idx, 1, 0] - pos_y) ** 2
                    coll_cost += coll_pos_prob[2, i] * (rho_nn[coll_idx, 0, 0] * distance_to_free_cell).sum()

                return "collision", idx, coll_cost
        return "success", x_nn, rho_nn

    def load_predictor(self, args, dim_x):
        _, num_inputs = load_inputmap(dim_x, args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True)
        model.eval()
        return model