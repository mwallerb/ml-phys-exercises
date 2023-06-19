# Convenience function for creating movies inside JupyterLab
#
# Copyright 2023 Markus Wallerberger
# SPDX-License-Identifier: MIT
import warnings
import numpy as np
import matplotlib.pyplot as pl
import imageio.v3 as imageio
import IPython.display as ipy_display
import gymnasium as gym


class Movie:
    """Create movie from environment"""
    def __init__(self):
        self.images = []

    def clear(self):
        """Destroy the movie"""
        self.images.clear()
    
    def add_state(self, env):
        """Capture the current state of a gym environment"""
        if env.render_mode != 'rgb_array':
            raise ValueError("Environment needs to be created with "
                             "render_mode='rgb_array' argument")
        self.images.append(env.render())
        
    def add_figure(self, fig=None):
        """Capture a matplotlib plot"""
        self.images.append(figure_to_rgb_array(fig))
        
    def create(self, filename, loops=0, frame_duration=20):
        """Create the captured movie"""
        if frame_duration < 20:
            warnings.warn("Frame durations < 20 ms seems to not work")
        durations = np.full(len(self.images), frame_duration)
        durations[0] = 250
        durations[-1] = 1000
        return imageio.imwrite(filename, self.images, extension=".gif", 
                               loop=loops, duration=list(durations))

    def show(self, loops=0, frame_duration=20):
        """Render the captured movie to the Jupyter notebook"""
        buffer = self.create("<bytes>", loops, frame_duration)
        return ipy_display.Image(buffer)

    
def figure_to_rgb_array(fig=None):
    """Convert a Matplotlib figure to a RGB array"""
    if fig is None:
        fig = pl.gcf()
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
