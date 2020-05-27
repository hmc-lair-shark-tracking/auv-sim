import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

# time step (unit: sec)
DELTA_T = 0.1

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 1.0

def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians
    """
    if -np.pi <= ang <= np.pi:
        return ang
    elif ang > np.pi: 
        ang += (-2 * np.pi)
        return angle_wrap(ang)
    elif ang < -np.pi: 
        ang += (2 * np.pi)
        return angle_wrap(ang)


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
            spaces.Box(low = np.array([auv_init_pos.x - 200.0, auv_init_pos.y - 200.0, -200.0, 0.0]), high = np.array([auv_init_pos.x + 200.0, auv_init_pos.y + 200.0, 0.0, 0.0]), dtype = np.float64),\
            spaces.Box(low = np.array([shark_init_pos.x - 200.0, shark_init_pos.y - 200.0, -200.0, 0.0]), high = np.array([shark_init_pos.x + 200.0, shark_init_pos.y + 200.0, 0.0, 0.0]), dtype = np.float64)))

        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos

        self.reset()


    def step(self, action):
        v, w = action
        
        x, y, z, theta = self.state[0]

        new_x = x + v * np.cos(theta) * DELTA_T
        new_y = y + v * np.sin(theta) * DELTA_T
        new_theta = angle_wrap(theta + w * DELTA_T)

        # TODO: For now, the shark's position does not change. Might get updated in the future 
        new_shark_pos = self.state[1]
        
        self.calculate_range(self.state[0], self.state[1])

        self.state = (np.array([new_x, new_y, z, new_theta]), new_shark_pos)

        range = self.calculate_range(self.state[0], self.state[1])

        done = self.check_end_eps(range)

        reward = self.get_reward()

        return self.state, reward, done, {}

    
    def calculate_range(self, auv_pos, shark_pos):
        auv_x = auv_pos[0]
        auv_y = auv_pos[1]
        shark_x = shark_pos[0]
        shark_y = shark_pos[1]

        delta_x = shark_x - auv_x
        delta_y = shark_y - auv_y

        return np.sqrt(delta_x**2 + delta_y**2)


    def check_end_eps(self, range):
        """
        Return:
            True, if the auv is within the end game region specified by the END_GAME_RADIUS
            False, if not
        """
        return  range <= END_GAME_RADIUS


    def get_reward(self):
        """
        Return the reward that the auv gets at this time step
        """
        return 0.0

    def reset(self):
        """
        Reset the environment
        """
        self.state = (np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta]),\
            np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z, self.shark_init_pos.theta]))
        return self.state

    def render(self, mode='human'):
        ...

    def close(self):
        ...
    