import numpy as np
import torch
from motion_planning.utils import pos2gridpos, gridpos2pos, plot_grid, shift_array


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
        self.grid = np.zeros((self.grid_size[0], self.grid_size[1], number_timesteps))
        for obj in self.objects:
            if obj.grid.shape[2] < number_timesteps:
                obj.forward_occupancy(step_size=number_timesteps - obj.grid.shape[2])
            self.add_grid(obj.grid)

    def add_grid(self, grid):
        """add individual object grid to the overall occupancy grid"""
        self.grid = np.clip(self.grid + grid[:, :, :self.current_timestep + 1], 0, 1)

    def add_object(self, obj):
        """add individual object to the environment and add occupancy grid"""
        if obj.grid_size == self.grid_size and isinstance(obj, OccupancyObject):
            self.objects.append(obj)
            self.add_grid(obj.grid)
        else:
            raise Exception("Gridsize must match and object must be of type OccupancyObject!")

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

    def __init__(self, args, name="staticObstacle", coord=None, timestep=0):
        self.grid_size = args.grid_size
        self.current_timestep = 0
        self.name = name
        self.grid = np.zeros((self.grid_size[0], self.grid_size[1], timestep + 1))
        if coord is not None:
            self.add_occupancy(args, pos=coord[0:4], certainty=coord[4], spread=coord[5], timestep=0)

    def add_occupancy(self, args, pos, certainty=1, spread=1, timestep=None):
        if timestep is None:
            timestep = self.current_timestep
        grid_pos_x, grid_pos_y = pos2gridpos(args, pos[:2], pos[2:])
        if self.grid.shape[2] < timestep:
            self.grid[:, :, timestep] = np.zeros(self.grid_size)
        for i in range(int(spread)):
            self.grid[grid_pos_x[0] - i:grid_pos_x[1] + i, grid_pos_y[0] - i:grid_pos_y[1] + i, timestep] \
                = self.grid[grid_pos_x[0] - i:grid_pos_x[1] + i, grid_pos_y[0] - i:grid_pos_y[1] + i, timestep] + certainty / spread

    def forward_occupancy(self, step_size=1):
        self.grid = np.concatenate((self.grid, np.repeat(self.grid[:, :, [self.current_timestep]], step_size,
                                                             axis=2)), axis=2)
        self.current_timestep += step_size

    def enlarge_shape(self):
        """enlarge obstacle by the shape of the ego vehicle"""

class DynamicObstacle(StaticObstacle):
    def __init__(self, args, name="dynamicObstacle", timestep=0, coord=None, velocity_x=0, velocity_y=0):
        super().__init__(args, name=name, coord=coord, timestep=timestep)
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def forward_occupancy(self, step_size=1):
        self.grid = np.concatenate((self.grid, np.zeros((self.grid_size[0], self.grid_size[1], step_size))), axis=2)
        for i in range(step_size):
            self.grid[:, :, self.current_timestep + 1 + i] = shift_array(self.grid[:, :, self.current_timestep + i], self.velocity_x, self.velocity_y)
        self.current_timestep += step_size


class EgoVehicle:
    # wide = 10

    def __init__(self, start, goal, args):
        self.start = start
        self.goal = goal
        self.predictor = self.predictor_wrapper(args)

    def predictor_wrapper(self, args):
        """
        Return neural pdf predictor
        """

        _predictor = self.load_predictor(args)

        def predictor(xref0, uref):
            pass
            # 1. sample batch xe0
            # 2. predict xe(t) and rho(t) for times t
                #= _predictor(xref0, uref, t, ...)
            # 3. interpolate xe(t), rho(t) to get pdf at t
            # 4. compute xref
            # 5. return grid with final occupation probabilities

        return predictor

    @staticmethod
    def load_predictor(args):
        """
        load the pretrained pdf predictor
        """

        _predictor = torch.load(args.name_pretrained_nn, map_location=torch.device('cpu'))
        _predictor.cpu()
        return _predictor
