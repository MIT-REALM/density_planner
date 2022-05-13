import matplotlib.pyplot as plt
import numpy as np

def plot_grid(object, args, timestep=None):
    if timestep is None:
        timestep = object.current_timestep
    x_wide = max((args.environment_size[1] - args.environment_size[0]) / 10, 3)
    y_wide = (args.environment_size[3] - args.environment_size[2]) / 10
    plt.figure(figsize=(x_wide, y_wide))
    plt.pcolormesh(object.grid[:, :, timestep].T, cmap='binary')
    plt.axis('scaled')

    ticks_x = np.concatenate((np.arange(0, args.environment_size[1]+1, 10), np.arange(-10, args.environment_size[0]-1, -10)), 0)
    ticks_y = np.concatenate((np.arange(0, args.environment_size[3]+1, 10), np.arange(-10, args.environment_size[2]-1, -10)), 0)
    ticks_x_grid, ticks_y_grid = pos2gridpos(args, ticks_x, ticks_y)
    plt.xticks(ticks_x_grid, ticks_x)
    plt.yticks(ticks_y_grid, ticks_y)

    plt.title(f"{object.name} at timestep={timestep}")
    plt.tight_layout()
    plt.show()


def pos2gridpos(args, pos_x=None, pos_y=None):
    if pos_x is not None:
        if isinstance(pos_x, list):
            pos_x = np.array(pos_x)
        pos_x = ((pos_x - args.environment_size[0]) / (args.environment_size[1] - args.environment_size[0])) * \
               args.grid_size[0]
    if pos_y is not None:
        if isinstance(pos_y, list):
            pos_y = np.array(pos_y)
        pos_y = ((pos_y - args.environment_size[2]) / (args.environment_size[3] - args.environment_size[2])) * \
               args.grid_size[1]
    return pos_x.astype(int), pos_y.astype(int)

def gridpos2pos(args, pos_x=None, pos_y=None):
    if pos_x is not None:
        pos_x = (pos_x / args.grid_size[0]) * (args.environment_size[1] - args.environment_size[0]) + \
            args.environment_size[0]
    if pos_y is not None:
        pos_y = (pos_y / args.grid_size[1]) * (args.environment_size[3] - args.environment_size[2]) + \
        args.environment_size[2]
    return pos_x, pos_y

def shift_array(arr, step_x=0, step_y=0):
    result = np.empty_like(arr)
    if step_x > 0:
        result[:step_x, :] = 0
        result[step_x:, :] = arr[:-step_x, :]
    elif step_x < 0:
        result[step_x:, :] = 0
        result[:step_x, :] = arr[-step_x:, :]
    arr = result
    if step_y > 0:
        result[:, step_y] = 0
        result[:, step_y:] = arr[:, -step_y]
    elif step_y < 0:
        result[:, step_y] = 0
        result[:, step_y] = arr[: -step_y:]
    else:
        result[:] = arr
    return result

#
# def create_example_grid():
#     gridsize = (100, 100)
#     ntimesteps = 20
#     obstacles = (
#         (70, 75, 40, 45, 0.8, 20, "Ped1"), (0, 20, 0, 100, 1, 1, "static1"),
#         (80, 100, 0, 100, 1, 1, "static2"))  # (x1, x2, y1, y2, certainty, spread)
#     grid = Environment(gridsize=gridsize)
#
#     for coord in obstacles:
#         obj = OccupancyObject(gridsize=gridsize, name=coord[6],  pos=coord[0:4], certainty=coord[4], spread=coord[5], timestep=0)
#         grid.add_object(obj)
#
#     grid.objects[1].forward_occupancy(0,5)
#     grid.objects[1].plot_grid(5)
#     grid.plot_grid(5)
#
#     return grid
#
# def create_example_vehicle():
#     start = (60, 2)
#     goal = (70, 100)
#     T = 0.2
#     vehicle = Vehicle(start, goal, T)
#
#     return vehicle