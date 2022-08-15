import os

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use
import matplotlib.animation as anim
import matplotlib.path as mpltPath
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, threshold_isodata, try_all_threshold

# adding submodules to the system path
sys.path.insert(0, './external/drone-dataset-tools/src/')
# noinspection PyUnresolvedReferences
from tracks_import import read_from_csv

sys.path.insert(0, '../motion_planning/')
# noinspection PyUnresolvedReferences
from simulation_objects import Environment, StaticObstacle, DynamicObstacle


def image2grid(image):
    # Convert image to grid by first applying a down sampling filter
    # and then applying a thresholding filter
    grid = np.zeros(image.shape)
    grid[image > 0.5] = 1
    return grid


def get_image_from_file(file_path):
    # Read image from file
    image = imread(file_path)
    return image


# Read image from file
image = get_image_from_file("./inD-dataset-v1.0/data/00_background.png")
# Convert image to grid by first applying a down sampling filter
# and then applying a thresholding filter
# grid = image2grid(image)
# Mask out the boundary of the image
gray_file = rgb2gray(image)  # convert to grayscale
downsampled_img = block_reduce(gray_file, block_size=(2, 2), func=np.mean)
threshold = threshold_isodata(downsampled_img)  # input value

# fig, ax = try_all_threshold(gray_file, figsize=(10, 8), verbose=False)
# plt.show()
binary = (downsampled_img < threshold)

# # plt.show image
# implt.show(downsampled_img, cmap='gray')
# plt.show()
# plt.imshow(binary, cmap='gray')
# plt.show()
# plt.imshow(image)
# plt.show()
# plt.imshow(image2grid(binary), cmap='gray')
# plt.show()

tracks, tracks_meta, recording_meta = read_from_csv(tracks_file="./inD-dataset-v1.0/data/00_tracks.csv",
                                                    tracks_meta_file="./inD-dataset-v1.0/data/00_tracksMeta.csv",
                                                    recording_meta_file="./inD-dataset-v1.0/data/00_recordingMeta.csv",
                                                    include_px_coordinates=True)

# Create environmental variables
STEP_SIZE = 1 / recording_meta['frameRate']  # in seconds
GRID_RESOLUTION = 0.5/12  # [m]
SIMULATION_TIME = recording_meta['duration']  # [s]
SCALING_FACTOR = recording_meta['orthoPxToMeter']  # [m/px]
NUM_VEHICLES = recording_meta['numVehicles']
NUM_VRUS = recording_meta['numVRUs']
NUM_OBSTACLES = recording_meta['numTracks']
NUM_FRAMES = int(np.round(SIMULATION_TIME / STEP_SIZE)) + 1

# Compute environment size
y_dim, x_dim = image.shape[:2]
x_dim = x_dim * SCALING_FACTOR
y_dim = y_dim * SCALING_FACTOR
environment_size = [0, x_dim, 0, y_dim]
M, N = (int((x_dim / GRID_RESOLUTION)), int((y_dim / GRID_RESOLUTION)))
# Intitialize figures.
map_fig = plt.figure()
map_ax = map_fig.add_subplot(111)
map_ax.set_xlim(0, M)
map_ax.set_ylim(0, N)

# Populate static obstacles and boundaries
x_scale = int(np.round(GRID_RESOLUTION / SCALING_FACTOR))
y_scale = int(np.round(GRID_RESOLUTION / SCALING_FACTOR))
background_grid = block_reduce(gray_file, block_size=(x_scale, y_scale), func=np.mean)
threshold = threshold_isodata(gray_file)  # input value
binary_grid = background_grid < threshold
occupancy_grid3D = np.repeat(binary_grid[:, :, np.newaxis], NUM_FRAMES, axis=2)

# create spacing points
x_pts = np.linspace(0, x_dim, binary_grid.shape[1])
y_pts = np.linspace(0, -y_dim, binary_grid.shape[0])
time_pts = np.linspace(0, SIMULATION_TIME, NUM_FRAMES)

occupancy_grid2D = np.meshgrid(x_pts, y_pts)  # make a canvas with coordinates
points = np.vstack((occupancy_grid2D[0].flatten(), occupancy_grid2D[1].flatten())).T


# Create interpolator
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin(keepdims=True)
    return idx, array[idx]


# Iterate over all tracks
for track in tracks:
    position = np.array([track['xCenter'], track['yCenter']]) / 12
    for indx in range(len(track['frame'])):
        frame = track['frame'][indx]
        if track["bbox"] is not None:
            bounding_box = track["bbox"][indx] / 12
            path = mpltPath.Path(bounding_box)
            inside_pts = path.contains_points(points)
            if inside_pts.any():
                inside_pts = points[inside_pts]
                for inside_pt in inside_pts:
                    x_indx = np.where(x_pts == inside_pt[0])[0]
                    y_indx = np.where(y_pts == inside_pt[1])[0]
                    occupancy_grid3D[y_indx, x_indx, frame] = 0.5
        else:
            x_indx = find_nearest_index(x_pts, position[0][indx])[0]
            y_indx = find_nearest_index(y_pts, position[1][indx])[0]
            # Populate occupancy grid
            occupancy_grid3D[y_indx, x_indx, frame] = 0.5


def map_update(i, occupancy_grid, ax):
    ax.clear()
    ax.set_xlim(0, M)
    ax.set_ylim(0, N)
    ax.imshow(occupancy_grid[:, :, i], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)


map_anim = anim.FuncAnimation(map_fig, map_update, frames=np.linspace(0, NUM_FRAMES - 1, dtype=int),
                              fargs=(occupancy_grid3D, map_ax), repeat=True)
f = r"./animation.mp4"
writervideo = anim.FFMpegWriter(fps=5)
map_anim.save(f, writer=writervideo)


print('done')
