import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class AuvEnv(gym.Env):
    # possible render modes: human, rgb_array (creates image for making videos), ansi (string)
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # action: 
        #   a tuple of (v, w), linear velocity and angular velocity
        # range for v (unit: m/s): [-2, 2]
        # range for w (unit: radians): [-pi, pi]
        #   TODO: Currently, the track_way_point function has K_P == 1, so this is the range for w. Might change in the future?
        self.action_space = spaces.Box(low = np.array([-2, -np.pi]), high = np.array([2, np.pi]), dtype = np.float32)

        # observation:
        #   the x, y, z position of the auv
        #   the x, y, z position of the shark (TODO: one shark for now?)
        #       range for x, y (unit: m): [-200, 200]
        #       range for z (unit: m): [-200, 0]
        self.observation_space = spaces.Tuple(\
            spaces.Box(low = np.array([-200, -200, -200]), high = np.array([200, 200, 0]), dtype = np.float32),\
            spaces.Box(low = np.array([-200, -200, -200), high = np.array([200, 200, 0]), dtype = np.float32))


    def step(self, action):
        ...

    def reset(self):
        ...
        
    def render(self, mode='human'):
        ...
    def close(self):
        ...