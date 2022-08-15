import numpy as np
import torch
import pygame
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.path as mplt_path

from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger
from skimage.io import imread
from skimage.measure import block_reduce
from skimage.color import rgb2gray
from skimage.util import crop
from skimage.filters import threshold_isodata
from motion_planning.utils import enlarge_grid, compute_gradient
from env.utils import Configurable

# adding submodules to the system path
sys.path.insert(0, './env/external/drone-dataset-tools/src/')
# noinspection PyUnresolvedReferences
from tracks_import import read_from_csv


class Environment(Configurable):
    """Combined grip map of multiple OccupancyObjects"""

    def __init__(self, name="environment", init_time=0, config=None):
        # Initialize superclass
        super().__init__()
        Configurable.__init__(self, config)
        self.map_anim = None
        self.name = name
        self.current_time = init_time
        # Extract tracks and records from config
        self.__extract_recording_data()
        # Initialize tracks
        self.__initialize_tracks(self.__tracks, self.__tracks_meta)
        # Initialize environmental parameters
        self.scale_down_factor = 12
        self.step_size = 1 / self.__recording_meta['frameRate']  # in seconds
        self.grid_resolution = self.config['grid']['resolution'] / self.scale_down_factor  # [m]
        self.simulation_time = self.__recording_meta['duration']  # [s]
        self.scaling_factor = self.__recording_meta['orthoPxToMeter']  # [m/px]
        self.num_vehicles = self.__recording_meta['numVehicles']
        self.num_vrus = self.__recording_meta['numVRUs']
        self.num_obstacles = self.__recording_meta['numTracks']
        self.num_frames = self.maximum_frame - self.minimum_frame + 1
        self._init_frame = int(np.round(init_time / self.step_size))
        self.current_frame = self._init_frame

        # Check if visualization should be included
        self.visualize = self.config['environment']['visualize']
        if self.visualize:
            # Initialize visualization
            self.__initialize_visualization()

        # Initialize grid with background image and static obstacles
        self.__create_grid()
        self.grid_enlarged = None
        self.grid_gradientX = None
        self.grid_gradientY = None

    # noinspection PyTypeChecker
    def run(self) -> torch.Tensor:
        # iterate frames and update the grid
        total_frames = self.num_frames - self.current_frame

        if self.config['environment']["parallel"]:
            logger.info("Iterating through {} frames on {} cores", total_frames, os.cpu_count())
            pool = Pool(os.cpu_count())
            inputs = tqdm(range(self.current_frame, self.num_frames))
            pool.imap_unordered(self.update_grid, inputs, chunksize=os.cpu_count())
            # close the pool and wait for the work to finish
            pool.close()
            pool.join()
            logger.info("Closing pool")
        else:
            logger.info("Iterating through {} frames", total_frames)
            for frame_idx in tqdm(range(self.current_frame, self.num_frames)):
                self.update_grid(frame_idx)
        logger.info("...Done")
        return self.grid

    def create_animation(self):
        if not self.visualize:
            raise Exception('Visualization is not enabled on configuration file.')
        """create the animation"""
        logger.info("Creating animation!")
        # Create animation
        # noinspection PyTypeChecker
        self.map_anim = anim.FuncAnimation(self.fig, self.__update_animation,
                                           frames=tqdm(np.arange(self._init_frame, self.num_frames - 1)),
                                           interval=1000 / self.config['environment']["fps"])
        # Save animation
        self.map_anim.save(self.output_file, writer=self.writer_video)
        logger.info("...Done!")
        # Close figure
        plt.close(self.fig)

    def __initialize_tracks(self, tracks, tracks_meta):
        """initialize the tracks settings"""
        # Check whether tracks and tracks_meta match each other
        error_message = "The tracks file and the tracksMeta file is not matching each other. " \
                        "Please check whether you modified any of these files."
        if len(tracks) != len(tracks_meta):
            logger.error(error_message)
            raise DataError("Failed", error_message)
        for track, track_meta in zip(tracks, tracks_meta):
            if track["trackId"] != track_meta["trackId"]:
                logger.error(error_message)
                raise DataError("Failed", error_message)
        # Determine the first and last frame
        self.minimum_frame = min(meta["initialFrame"] for meta in tracks_meta)
        self.maximum_frame = max(meta["finalFrame"] for meta in tracks_meta)
        logger.info("The recording contains tracks from frame {} to {}.", self.minimum_frame, self.maximum_frame)
        # Create a mapping between frame and idxs of tracks for quick lookup during playback
        self.frame_to_track_idxs = {}
        for i_frame in range(self.minimum_frame, self.maximum_frame + 1):
            indices = [i_track for i_track, track_meta in enumerate(tracks_meta)
                       if track_meta["initialFrame"] <= i_frame <= track_meta["finalFrame"]]
            self.frame_to_track_idxs[i_frame] = indices

    def __initialize_visualization(self):
        """initialize the visualization settings and window"""
        # Create figure and axes
        self.fig, self.ax = plt.subplots(1, 1)
        self.fig.set_size_inches(15, 8)
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.10, top=1.00)
        # Create output folder
        self.output_file = self.config['environment']["output_file"]
        self.writer_video = anim.FFMpegWriter(fps=self.config['environment']["fps"])

    def __create_grid(self):
        """creates an occupancy grid based on background image"""
        # Compute environment size
        left, upper, right, lower = np.array(self.cropped_dims) * self.scaling_factor
        x_dim = right - left  # [m]
        y_dim = -(lower - upper)  # [m]
        # compute environment size
        self.grid_size = np.array([x_dim, y_dim]) * self.scale_down_factor  # [m]
        self.grid_size_px = [int((x_dim / self.grid_resolution)), int((-y_dim / self.grid_resolution))]  # [px]
        # Populate static obstacles and road boundaries
        x_scale = int(np.round(self.grid_resolution / self.scaling_factor))
        y_scale = int(np.round(self.grid_resolution / self.scaling_factor))
        logger.info("Rescaling image with pool of size {}.", np.array([x_scale, y_scale]))
        background_grid = block_reduce(self.background_image_cropped, block_size=(x_scale, y_scale), func=np.mean)
        threshold = threshold_isodata(self.background_image)  # input value
        # Create binary occupancy grid
        binary_grid = background_grid < threshold
        number_time_steps = int(self.num_frames - self.current_frame)
        # Create occupancy grid for each time step
        self.grid = torch.tensor(np.repeat(binary_grid[:, :, np.newaxis], number_time_steps, axis=2))
        # Create spacing points for the grid
        self.__x_pts = np.linspace(left, right, binary_grid.shape[1])
        self.__y_pts = np.linspace(-upper, -lower, binary_grid.shape[0])
        self.__time_pts = np.linspace(self._init_frame, self.num_frames, number_time_steps)

        grid_2d = np.meshgrid(self.__x_pts, self.__y_pts)  # make a canvas with coordinates
        self.__points = np.vstack((grid_2d[0].flatten(), grid_2d[1].flatten())).T

    def __extract_recording_data(self):
        """extracts the recordings from config"""
        # Extract data from recording
        dataset_dir = self.config['dataset']["dataset_dir"]
        if dataset_dir and os.path.exists(dataset_dir):
            # Generate file based on recording number
            recording = self.config['dataset']["recording"]
            if recording is None:
                raise Exception("Recording number not specified!")

            recording = "{:02d}".format(int(recording))
            # Create paths to csv files
            tracks_file = dataset_dir + recording + "_tracks.csv"
            tracks_meta_file = dataset_dir + recording + "_tracksMeta.csv"
            recording_meta_file = dataset_dir + recording + "_recordingMeta.csv"
            # Load csv files
            self.__tracks, self.__tracks_meta, self.__recording_meta = \
                read_from_csv(tracks_file=tracks_file,
                              tracks_meta_file=tracks_meta_file,
                              recording_meta_file=recording_meta_file)
            # Bring image background
            background_image_path = dataset_dir + recording + "_background.png"
            if background_image_path and os.path.exists(background_image_path):
                logger.info("Loading background image from {}", background_image_path)
                self.background_image = rgb2gray(imread(background_image_path))
                (self.image_height, self.image_width) = self.background_image.shape
                # crop background image
                self.background_image_cropped, self.cropped_dims = self.crop_image_interactively(background_image_path)
            else:
                raise Exception("Background image not found!")
        else:
            raise Exception("Invalid dataset path!")

    def update_grid(self, frame_idx) -> torch.Tensor:
        """
        Main function to update the grid with the current time step. It updates the current time step by one
        and updates the grid with the current time step.

        param: frame_idx: index of the frame to be updated in the grid
        :return: torch tensor with current occupancy grid in the shape of (x_dim, y_dim)
        """
        # Iterate through all tracks in current frame and update the grid
        for track_idx in self.frame_to_track_idxs[frame_idx]:
            # Get track data
            track = self.__tracks[track_idx]

            # track_id = track["trackId"]
            track_meta = self.__tracks_meta[track_idx]
            initial_frame = track_meta["initialFrame"]

            current_index = frame_idx - initial_frame

            # Instantiate obstacle on grid
            # Vehicles are represented as polygons
            if track["bbox"] is not None:
                # extract obstacle bounding box
                bounding_box = track["bbox"][current_index] / self.scale_down_factor
                path = mplt_path.Path(bounding_box)
                inside_pts = path.contains_points(self.__points)
                if inside_pts.any():
                    inside_pts = self.__points[inside_pts]
                    for inside_pt in inside_pts:
                        x_indx = np.where(self.__x_pts == inside_pt[0])[0]
                        y_indx = np.where(self.__y_pts == inside_pt[1])[0]
                        # Populate grid with obstacle
                        self.grid[y_indx, x_indx, frame_idx] = 1
                        """ NOTE: Here is where uncertainty can be added """
            # VRUs are represented as dots
            else:
                # extract obstacle centroid position
                x_position = track['xCenter'][current_index] / self.scale_down_factor
                y_position = track['yCenter'][current_index] / self.scale_down_factor
                # Compute nearest grid point
                x_indx = self.find_nearest_index(self.__x_pts, x_position)[0]
                y_indx = self.find_nearest_index(self.__y_pts, y_position)[0]
                # Populate occupancy grid
                self.grid[y_indx, x_indx, frame_idx] = 1
                """ NOTE: Here is where uncertainty can be added """
        # time.sleep(0.2)
        return self.grid[:, :, frame_idx]

    def enlarge_shape(self, table=None):
        """enlarge the shape of all obstacles and update the grid to do motion planning for a point"""
        if table is None:
            table = [[0, 10, 25], [10, 30, 20], [30, 50, 10], [50, 101, 5]]
        grid_enlarged = self.grid.clone().detach()
        for elements in table:
            if elements[0] >= self.grid.shape[2]:
                continue
            time_steps = torch.arange(elements[0], min(elements[1], self.grid.shape[2]))
            grid_enlarged[:, :, time_steps] = enlarge_grid(self.grid[:, :, time_steps], elements[2])
        self.grid_enlarged = grid_enlarged

    def get_gradient(self):
        if self.grid_gradientX is None:
            grid_gradient_x, grid_gradient_y = compute_gradient(self.grid, step=1)
            s = 5
            missing_grad = torch.logical_and(self.grid != 0,
                                             torch.logical_and(grid_gradient_x == 0, grid_gradient_y == 0))
            while torch.any(missing_grad):
                idx = missing_grad.nonzero(as_tuple=True)
                grid_gradient_x_new, grid_gradient_y_new = compute_gradient(self.grid, step=s)
                grid_gradient_x[idx] += s * grid_gradient_x_new[idx]
                grid_gradient_y[idx] += s * grid_gradient_y_new[idx]
                s += 10
                missing_grad = torch.logical_and(self.grid != 0,
                                                 torch.logical_and(grid_gradient_x == 0, grid_gradient_y == 0))

            self.grid_gradientX = grid_gradient_x
            self.grid_gradientY = grid_gradient_y

    def __update_animation(self, frame):
        """
        Update the animation with the current time step. It updates the current time step by one
        and updates the animation with the current time step.
        :param frame: current frame
        :param grid: current occupancy grid
        :param ax: current axis
        :return: None
        """
        self.ax.clear()
        self.ax.set_xlim(0, self.grid.shape[1])
        self.ax.set_ylim(0, self.grid.shape[0])
        self.ax.imshow(self.grid[:, :, frame], cmap='gray', origin='lower', vmin=0.0, vmax=1.0)

    @staticmethod
    # Create interpolator
    def find_nearest_index(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin(keepdims=True)
        return idx  # , array[idx]

    @staticmethod
    def __display_image(screen, px, top_left, prior):
        """
        Display the image on the screen.
        :param screen: screen to display the image
        :param px: image to display
        :param top_left: top left corner of the image
        :param prior: prior of the image
        :return: None
        """
        # ensure that the rect always has positive width, height
        x, y = top_left
        width = pygame.mouse.get_pos()[0] - top_left[0]
        height = pygame.mouse.get_pos()[1] - top_left[1]
        if width < 0:
            x += width
            width = abs(width)
        if height < 0:
            y += height
            height = abs(height)

        # eliminate redundant drawing cycles (when mouse isn't moving)
        current = x, y, width, height
        if not (width and height):
            return current
        if current == prior:
            return current

        # draw transparent box and blit it onto canvas
        screen.blit(px, px.get_rect())
        im = pygame.Surface((width, height))
        im.fill((128, 128, 128))
        pygame.draw.rect(im, (32, 32, 32), im.get_rect(), 1)
        im.set_alpha(128)
        screen.blit(im, (x, y))
        pygame.display.flip()

        # return current box extents
        return x, y, width, height

    @staticmethod
    def __setup_display(path):
        """
        Setup the display.
        :param screen: screen to display the image
        :return: None
        """
        px = pygame.image.load(path)
        screen = pygame.display.set_mode(px.get_rect()[2:])
        screen.blit(px, px.get_rect())
        pygame.display.flip()
        return screen, px

    def __interactive_image_crop(self, screen, px):
        """
        Interactive image crop.
        :param screen: screen to display the image
        :param px: image to display
        :return: None
        """
        top_left = bottom_right = prior = None
        n = 0
        while n != 1:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if not top_left:
                        top_left = event.pos
                    else:
                        bottom_right = event.pos
                        n = 1
            if top_left:
                prior = self.__display_image(screen, px, top_left, prior)
        pygame.display.quit()
        # noinspection PyUnreachableCode
        return top_left + bottom_right

    def crop_image_interactively(self, path):
        """
        Crop the image interactively.
        :param path: path to the image
        :return: cropped image, top left corner, bottom right corner
        """
        screen, px = self.__setup_display(path)
        left, upper, right, lower = self.__interactive_image_crop(screen, px)
        # ensure output rect always has positive width, height
        if right < left:
            left, right = right, left
        if lower < upper:
            lower, upper = upper, lower
        # crop the image
        img = rgb2gray(imread(path))[upper:lower, left:right]
        return img, (left, upper, right, lower)

    @classmethod
    def default_config(cls):
        return dict(dataset=dict(
            dataset_dir="./env/inD-dataset-v1.0/data/",
            recording=1,
        ),
            grid=dict(
                resolution=0.5,
                certainty=np.random.randint(3, 10) / 10,
                spread=np.random.randint(1, 30)
            ),
            environment=dict(
                visualize=True,
                parallel=True,
                output_file=r"./env/animation.mp4",
                fps=10
            )
        )


class DataError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
