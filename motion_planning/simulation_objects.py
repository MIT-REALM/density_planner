import numpy as np
import torch
from motion_planning.utils import pos2gridpos, check_collision, gridpos2pos, traj2grid, shift_array, pred2grid, get_mesh_sample_points, time2index, sample_pdf
from density_training.utils import load_nn, get_nn_prediction
from data_generation.utils import load_inputmap, load_outputmap
from plots.plot_functions import plot_ref
from plots.plot_functions import plot_grid

class Environment:
    """Combined grip map of multiple OccupancyObjects"""

    def __init__(self, objects, args, name="environment", timestep=0):
        # self.time = time
        self.grid_size = args.grid_size
        self.current_timestep = timestep
        self.objects = objects
        self.update_grid()
        self.name = name

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

    def enlarge_shape(self, shape, timestep=None):
        """enlarge the shape of all obstacles and update the grid to do motion planning for a point"""
        if timestep is None:
            timestep = self.current_timestep
        for obj in self.objects:
            obj.enlarge_shape(shape, timestep)
        self.update_grid()


class StaticObstacle:
    """Grid map of certain object with predicted occupancy probability value for each cell and timestep"""

    def __init__(self, args, coord, name="staticObstacle", timestep=0):
        self.grid_size = args.grid_size
        self.current_timestep = 0
        self.name = name
        self.grid = torch.zeros((self.grid_size[0], self.grid_size[1], timestep + 1))
        self.bounds = [None for _ in range(timestep + 1)]
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

    def enlarge_shape(self):
        """enlarge obstacle by the shape of the ego vehicle"""
        raise(NotImplementedError)


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

    def check_uref(self, uref_traj, xref_traj, i):
        distance_to_goal_old = ((xref_traj[0, :2, -1] - self.ego.xrefN[0, :2, 0]) ** 2).sum()

        # 1. compute xref
        xref_traj = self.ego.system.extend_xref_traj(xref_traj, uref_traj, self.args.dt_sim)
        if xref_traj is None:
            return "invalid"

        # 2. if collision or distance to goal from xrefN bigger than from xref0:
        #       return "invalid"
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
            if float(distance_to_goal) < 100:
                self.valid_traj.append([uref_traj, xref_traj, cost])
            return

        # 4. extend traj with u0 and u1
        for u0 in self.u0:
            for u1 in self.u1:
                u_ext = torch.tensor([u0, u1]).reshape((1, -1, 1)).repeat(1, 1, self.num_repeats)
                uref_traj_new = torch.cat((uref_traj, u_ext), dim=2)

                # 5. call check_uref
                self.check_uref(uref_traj_new, xref_traj, i)

    def check_collision_in_interv(self, xref_traj, i):
        timestep = i * self.num_repeats
        for j in range(self.num_repeats // self.args.factor_pred):
            interv = torch.arange(timestep, timestep + self.args.factor_pred)
            grid_xref = traj2grid(xref_traj[:, :, interv], self.args)
            collision, _ = check_collision(grid_xref, self.ego.grid_env[:, :, timestep // self.args.factor_pred], self.args.coll_sum)
            if collision:
                return True
            timestep += self.args.factor_pred
        return False


class EgoVehicle:
    def __init__(self, xref0, xrefN, pdf0, t_vec, system, env, args, name="egoVehicle"):
        self.xref0 = xref0
        self.xrefN = xrefN
        self.t_vec = t_vec
        self.system = system
        self.name = name
        self.grid_env = env.grid
        self.initialize_predictor(pdf0, args)

    def predict_cost(self, args, u_params):
        uref_traj, xref_traj = self.system.u_params2ref_traj(self.xref0, u_params, args)
        results, p0, p1 = self.predict_collision(u_params, xref_traj, args)
        if results == 'success':
            cost = (uref_traj ** 2).mean()
        else:
            cost = 1e10
        return cost

    def visualize_xref(self, xref_traj, args, uref_traj=None):
        if uref_traj is not None:
            plot_ref(xref_traj, uref_traj, 'Reference Trajectory', args, self.system, t=self.t_vec, include_date=True)
        grid = traj2grid(xref_traj, args)
        grid_env_max, _ = self.grid_env.max(dim=2)
        plot_grid(torch.clamp(grid + grid_env_max, 0, 1),  args, name='Reference Trajectory in the Environment')

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

    def predict_collision(self, u_params, xref_traj, args):
        # 2. predict x(t) and rho(t) for times t
        for t in self.t_vec:
            idx = time2index(t, args)
            xe_nn, rho_nn = get_nn_prediction(self.model, self.xe0, self.xref0, t, u_params, args)
            x_nn = xe_nn + xref_traj[:, :, [idx]]
            rho_nn *= self.rho0.reshape(-1, 1, 1)

            # 3. compute marginalized density grid
            grid_nn = pred2grid(x_nn[:, :, [0]], rho_nn, args)

            # 4. test for collisions
            collision, coll_sum = check_collision(grid_nn[:, :, 0], self.grid_env[:, :, idx], args.coll_sum)
            if collision:  # if coll_max > args.coll_max:
                return "collision", idx, coll_sum  # return "coll_max", idx, coll_max
        return "success", x_nn, rho_nn

    def load_predictor(self, args, dim_x):
        _, num_inputs = load_inputmap(dim_x, args)
        _, num_outputs = load_outputmap(dim_x)
        model, _ = load_nn(num_inputs, num_outputs, args, load_pretrained=True)
        model.eval()
        return model