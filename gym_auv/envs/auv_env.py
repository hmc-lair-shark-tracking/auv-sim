import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from live3DGraph import Live3DGraph

# size of the observation space
# the coordinates of the observation space will be based on 
#   the ENV_SIZE and the inital position of auv and the shark
# (unit: m)
ENV_SIZE = 500.0

# auv's max speed (unit: m/s)
AUV_MAX_V = 1.0
# auv's max angular velocity (unit: rad/s)
#   TODO: Currently, the track_way_point function has K_P == 1, so this is the range for w. Might change in the future?
AUV_MAX_W = np.pi/2

# time step (unit: sec)
DELTA_T = 1

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 1.0

# constants for reward
R_COLLIDE = -1000.0       # when the auv collides with an obstacle
R_ARRIVE = 1000.0         # when the auv arrives at the target
R_RANGE = 0.1           # this is a scaler to help determine immediate reward at a time step
R_AWAY = 0.1

def angle_wrap(ang):
    """
    Takes an angle in radians & sets it between the range of -pi to pi

    Parameter:
        ang - floating point number, angle in radians

    Note: 
        Because Python does not encourage importing files from the parent module, we have to place this angle wrap here. If we don't want to do this, we can possibly organize this so auv_env is in the parent folder?
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
        Declare the data members without initialize them
            automatically called when we build an environment with gym.make('gym_auv:auv-v0')

        Warning: 
            Need to immediately call init_env function to actually initialize the environment
        """
        self.action_space = None
        self.observation_space = None

        self.auv_init_pos = None
        self.shark_init_pos = None

        # the current state that the env is in
        self.state = None

        self.obstacle_array = []
        self.obstacle_array_for_rendering = []

        self.live_graph = Live3DGraph()

        self.auv_x_array_rl = []
        self.auv_y_array_rl = []
        self.auv_z_array_rl = []

        self.shark_x_array_rl = []
        self.shark_y_array_rl = []
        self.shark_z_array_rl = []


    def init_env(self, auv_init_pos, shark_init_pos, obstacle_array = []):
        """
        Initialize the environment based on the auv and shark's initial position

        Parameters:
            auv_init_pos - an motion plan state object
            shark_init_pos - an motion plan state object
            obstacle_array - 
        """
        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos
        self.obstacle_array_for_rendering = obstacle_array

        self.obstacle_array = []
        for obs in obstacle_array:
            self.obstacle_array.append([obs.x, obs.y, obs.z, obs.size])
        self.obstacle_array = np.array(self.obstacle_array)

        # action: 
        #   a tuple of (v, w), linear velocity and angular velocity
        # range for v (unit: m/s): [-AUV_MAX_V, AUV_MAX_V]
        # range for w (unit: radians): [-AUV_MAX_W, AUV_MAX_W]
        self.action_space = spaces.Box(low = np.array([-AUV_MAX_V, -AUV_MAX_W]), high = np.array([AUV_MAX_V, AUV_MAX_W]), dtype = np.float64)

        # observation: a tuple of 2 elements
        #   1. np array representing the auv's 
        #      [x_pos, y_pos, z_pos, theta]
        #   2. np array represent the shark's (TODO: one shark for now?)
        #      [x_pos, y_pos, z_pos, theta]
        self.observation_space = spaces.Box(low = np.array([auv_init_pos.x - ENV_SIZE, auv_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([auv_init_pos.x + ENV_SIZE, auv_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64)

        self.init_data_for_3D_plot(auv_init_pos, shark_init_pos)
        
        self.reset()


    def actions_range(self, N):
        v_options = np.linspace(-AUV_MAX_V, AUV_MAX_V, N)
        w_options = np.linspace(-AUV_MAX_W, AUV_MAX_W, N)
        w_options[0] = 0
        print((v_options, w_options))
        text = input("stop")
        # w_options = [0] * N
        return (v_options, w_options)


    def step(self, action):
        """
        Run one time step (defined as DELTA_T) of the dynamics in the environment

        Parameter:
            action - a tuple, representing the linear velocity and angular velocity of the auv

        Return:
            observation - a tuple of 2 np array, representing the auv and shark's new position
                each array has the format: [x_pos, y_pos, z_pos, theta]
            reward - float, amount of reward returned after previous action
            done - float, whether the episode has ended
            info - dictionary, can provide debugging info (TODO: right now, it's just an empty one)
        """
        v, w = action
        
        # get the old position and orientation data for the auv
        x, y, z, theta = self.state[0]
        # print("old position: ")
        # print(x, ", ", y, ", ", z, ", ", theta)

        old_range = self.calculate_range(self.state[0], self.state[1])

        # calculate the new position and orientation of the auv
        new_x = x + v * np.cos(theta) * DELTA_T
        new_y = y + v * np.sin(theta) * DELTA_T
        new_theta = angle_wrap(theta + w * DELTA_T)
        # print("new position: ")
        # print(new_x, ", ", new_y, ", ", z, ", ", new_theta)

        # TODO: For now, the shark's position does not change. Might get updated in the future 
        new_shark_pos = self.state[1]
        
        # update the current state to the new state
        self.state = (np.array([new_x, new_y, z, new_theta]), new_shark_pos, self.obstacle_array)

        # the episode will only end (done = True) if
        #   - the auv has reached the target, or
        #   - the auv has hit an obstacle
        done = self.check_reached_target(self.state[0], self.state[1]) or\
            self.check_collision(self.state[0])

        # reward = self.get_reward(old_range, self.state[0], self.state[1])
        reward = self.get_binary_reward(self.state[0], self.state[1])

        return self.state, reward, done, {}

    
    def calculate_range(self, a_pos, b_pos):
        """
        Calculate the range (distance) between point a and b, specified by their coordinates

        Parameters:
            a_pos - an array / a numpy array
            b_pos - an array / a numpy array
                both have the format: [x_pos, y_pos, z_pos, theta]

        TODO: include z pos in future range calculation?
        """
        a_x = a_pos[0]
        a_y = a_pos[1]
        b_x = b_pos[0]
        b_y = b_pos[1]

        delta_x = b_x - a_x
        delta_y = b_y - a_y

        return np.sqrt(delta_x**2 + delta_y**2)


    def check_reached_target(self, auv_pos, shark_pos):
        """
        Return:
            True, if the auv is within the end game region specified by the END_GAME_RADIUS
            False, if not
        """
        auv_shark_range = self.calculate_range(auv_pos, shark_pos)
        if auv_shark_range <= END_GAME_RADIUS:
            # print("Reached the Goal")
            return True
        else:
            return False


    def get_reward(self, old_range, auv_pos, shark_pos):
        """
        Return the reward that the auv gets at this state
        Specifically,
            if auv reaches the shark, R_ARRIVE
            if auv hits any obstacle, R_COLLIDE
            else,
                R_NEG + (immediate reward if auv gets closer to the shark)
        """
        if self.check_reached_target(auv_pos, shark_pos):
            return R_ARRIVE
        elif self.check_collision(auv_pos):
            return R_COLLIDE
        else:
            new_range = self.calculate_range(auv_pos, shark_pos)
            # if auv has gotten closer to the shark, will receive positive reward
            #   else, receive negative reward
            range_diff = old_range - new_range
            if range_diff <= 0: 
                reward = R_AWAY * range_diff
            else:
                reward = R_RANGE * range_diff
            return reward


    def get_binary_reward(self, auv_pos, goal_pos):
        """
        Return a binary reward (to help implementing HER algorithm in replay memory)
        Specifically,
            if auv reaches the shark, 1
            else, -1
        """
        if self.check_reached_target(auv_pos, goal_pos):
            return 1
        else:
            return -1
    

    def check_collision(self, auv_pos):
        """
        Check if the auv at the current state is hitting any obstacles

        Parameter:
            auv_pos - a np array [x, y, z, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= obs[3]:
                print("Hit an obstacle")
                return True
        return False


    def reset(self):
        """
        Reset the environment
            Set the observation to the initial auv and shark position
        """
        self.state = (np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta]),\
            np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z, self.shark_init_pos.theta]), self.obstacle_array)
        return self.state


    def render(self, mode='human'):
        """
        Render the environment by
            - printing out auv and shark's current position
            - returning self.state , so that another helper function can use live3DGraph.py class to plot the environment

        Return:
            a tuple of 2 np.array representing the auv and shark's current position
        """
        auv_pos = self.state[0]
        shark_pos = self.state[1]
        
        print("==========================")
        print("auv position: ")
        print("x = ", auv_pos[0], " y = ", auv_pos[1], " z = ", auv_pos[2], " theta = ", auv_pos[3])
        print("shark position: ")
        print("x = ", shark_pos[0], " y = ", shark_pos[1], " z = ", shark_pos[2], " theta = ", shark_pos[3])
        print("==========================")

        # self.render_3D_plot(self.state[0], self.state[1])
        return self.state


    def render_3D_plot(self, auv_pos, shark_pos):
        self.auv_x_array_rl.append(auv_pos[0])
        self.auv_y_array_rl.append(auv_pos[1])
        self.auv_z_array_rl.append(auv_pos[2])

        self.shark_x_array_rl.append(shark_pos[0])
        self.shark_y_array_rl.append(shark_pos[1])
        self.shark_z_array_rl.append(shark_pos[2])

        self.live_graph.plot_entity(self.auv_x_array_rl, self.auv_y_array_rl, self.auv_z_array_rl, label = 'auv', color = 'r', marker = ',')

        self.live_graph.plot_entity(self.shark_x_array_rl, self.shark_y_array_rl, self.shark_z_array_rl, label = 'shark', color = 'b', marker = 'o')

        if self.obstacle_array_for_rendering != []:
            self.live_graph.plot_obstacles(self.obstacle_array_for_rendering)

        self.live_graph.ax.legend()
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.001)

        self.live_graph.ax.clear()


    def init_data_for_3D_plot(self, auv_init_pos, shark_init_pos):
        self.auv_x_array_rl = [auv_init_pos.x]
        self.auv_y_array_rl = [auv_init_pos.y]
        self.auv_z_array_rl = [auv_init_pos.z]

        self.shark_x_array_rl = [shark_init_pos.x]
        self.shark_y_array_rl = [shark_init_pos.y]
        self.shark_z_array_rl = [shark_init_pos.z]


    