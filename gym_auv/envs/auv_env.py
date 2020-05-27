import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class AuvEnv(gym.Env):
    # possible render modes: human, rgb_array (creates image for making videos), ansi (string)
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Initialize the data members

        Warning: 
            Need to immediately call init_env function to actually initialize the environment
        """
        self.action_space = None
        self.observation_space = None

        self.auv_init_pos = None
        self.shark_init_pos = None

        # the current state that the env is in
        self.state = None


    def init_env(self, auv_init_pos, shark_init_pos):
        # action: 
        #   a tuple of (v, w), linear velocity and angular velocity
        # range for v (unit: m/s): [-2, 2]
        # range for w (unit: radians): [-pi, pi]
        #   TODO: Currently, the track_way_point function has K_P == 1, so this is the range for w. Might change in the future?
        self.action_space = spaces.Box(low = np.array([-2.0, -np.pi]), high = np.array([2.0, np.pi]), dtype = np.float64)

        # observation:
        #   the x, y, z position of the auv
        #   the x, y, z position of the shark (TODO: one shark for now?)
        #       range for x, y (unit: m): [-200, 200]
        #       range for z (unit: m): [-200, 0]
        self.observation_space = spaces.Tuple((\
            spaces.Box(low = np.array([auv_init_pos.x - 200.0, auv_init_pos.y - 200.0, -200.0]), high = np.array([auv_init_pos.x + 200.0, auv_init_pos.y + 200.0, 0.0]), dtype = np.float64),\
            spaces.Box(low = np.array([shark_init_pos.x - 200.0, shark_init_pos.y - 200.0, -200.0]), high = np.array([shark_init_pos.x + 200.0, shark_init_pos.y + 200.0, 0.0]), dtype = np.float64)))

        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos


    def step(self, action):
        ...

    def reset(self):
        """
        Reset the environment
        """
        self.state = (np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z]),\
            np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z]))
        return self.state

    def render(self, mode='human'):
        ...

    def close(self):
        ...
    