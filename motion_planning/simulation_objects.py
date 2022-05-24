import numpy as np
import torch
from motion_planning.utils import pos2gridpos, gridpos2pos, plot_grid, shift_array, get_pdf_grid, pdf2grid, get_mesh_sample_points


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
        self.grid = torch.zeros((self.grid_size[0], self.grid_size[1], timestep + 1))
        if coord is not None:
            self.add_occupancy(args, pos=coord[0:4], certainty=coord[4], spread=coord[5], timestep=0)

    def add_occupancy(self, args, pos, certainty=1, spread=1, timestep=None, pdf_form='square'):
        if timestep is None:
            timestep = self.current_timestep
        grid_pos_x, grid_pos_y = pos2gridpos(args, pos[:2], pos[2:])
        if self.grid.shape[2] < timestep:
            self.grid[:, :, timestep] = torch.zeros(self.grid_size)
        self.grid[:, :, timestep] = get_pdf_grid(self.grid[:, :, timestep], grid_pos_x, grid_pos_y, certainty, spread, pdf_form)

    def forward_occupancy(self, step_size=1):
        self.grid = torch.cat((self.grid, torch.repeat_interleave(self.grid[:, :, [self.current_timestep]], step_size,
                                                             dim=2)), dim=2)
        self.current_timestep += step_size

    def enlarge_shape(self):
        """enlarge obstacle by the shape of the ego vehicle"""
        raise(NotImplementedError)


class DynamicObstacle(StaticObstacle):
    def __init__(self, args, name="dynamicObstacle", timestep=0, coord=None, velocity_x=0, velocity_y=0):
        super().__init__(args, name=name, coord=coord, timestep=timestep)
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def forward_occupancy(self, step_size=1):
        self.grid = torch.cat((self.grid, torch.zeros((self.grid_size[0], self.grid_size[1], step_size))), dim=2)
        for i in range(step_size):
            self.grid[:, :, self.current_timestep + 1 + i] = shift_array(self.grid[:, :, self.current_timestep + i], self.velocity_x, self.velocity_y)
        self.current_timestep += step_size


class EgoVehicle:
    # wide = 10

    def __init__(self, system, args, start, goal, pdf0, name="egoVehicle"):
        self.system = system
        self.xref0 = start
        self.pdf0 = pdf0
        self.goal = goal
        self.name = name
        #self.predictor = self.predictor_wrapper(args)
        self.grid_size = args.grid_size

    def set_start_grid(self, args): #certainty=None, spread=2, pdf_form='squared'):
        self.grid = pdf2grid(self.pdf0, self.xref0, self.system, args)


    def predictor_wrapper(self, args):
        """
        Return neural pdf predictor
        """

        _predictor = self.load_predictor(args)



        return predictor

    @staticmethod
    def load_predictor(args):
        """
        load the pretrained pdf predictor
        """

        _predictor = torch.load(args.name_pretrained_nn, map_location=torch.device('cpu'))
        _predictor.cpu()
        return _predictor
