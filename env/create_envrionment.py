import cv2
import argparse
import os

import sys
import numpy as np
from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import threshold_ots
import json

from env.utils import Configurable

# adding submodules to the system path
sys.path.insert(0, './env/external/drone-dataset-tools/src')
from tracks_import import read_from_csv

from env.external import read_from_csv
from motion_planning.simulation_objects import Environment, StaticObstacle, DynamicObstacle


class IndEnvironment(Configurable):
    def __init__(self, config, object_str_list=None, name="environment", timestep=0):
        # Initialize superclass
        super().__init__()
        Configurable.__init__(self, config)

        self.args = args
        self.objects = []

        self.environment = Environment(self.objects, args, name=name)
        if timestep > 0:
            self.environment.forward_occupancy(step_size=timestep)
        self.timestep = timestep
        self.name = name

    @classmethod
    def default_config(cls):
        return dict(environment=dict(
            dataset="inD",
            dataset_dir="./env/inD-dataset-v1.0/data/",
            recording=26
        ),
            grid=dict(
                resolution=0.1,
                origin=[0, 0],
                occupancy_threshold=0.5,
                certainty=np.random.randint(3, 10) / 10,
                spread=np.random.randint(1, 30)
            )
        )

    def __image2grid(self, image):
        # Convert image to grid by first applying a down sampling filter
        # and then applying a thresholding filter


        grid = np.zeros(image.shape)
        grid[image > self.config["grid"]["occupancy_threshold"]] = 1
        return grid
    def __mask_out_boundary(self, image):
        # Mask out the boundary of the image
        gray_file = rgb2gray(image)  # convert to grayscale
        threshold = threshold_ots(gray_file)  # input value
        binary_file = (gray_file < threshold)



