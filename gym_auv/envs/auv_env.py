import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
import mpl_toolkits.mplot3d.art3d as Art3d
import copy
import random

from gym_auv.envs.live3DGraph_auv_env import Live3DGraph


# auv's max speed (unit: m/s)
AUV_MAX_V = 2.0
AUV_MIN_V = 0.1
# auv's max angular velocity (unit: rad/s)
#   TODO: Currently, the track_way_point function has K_P == 1, so this is the range for w. Might change in the future?
AUV_MAX_W = np.pi/8

# shark's speed (unit: m/s)
SHARK_MIN_V = 0.5
SHARK_MAX_V = 1
SHARK_MAX_W = np.pi/8

# time step (unit: sec)
DELTA_T = 0.1
# in step() funciton, determine how many times should an actino gets repeated
# so technically, the auv has moved for 0.5 sec in each loop
REPEAT_ACTION_TIME = 5

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 3.0
FOLLOWING_RADIUS = 50.0

# the auv will receive an immediate negative reward if it is close to the obstacles
OBSTACLE_ZONE = 3.0
WALL_ZONE = 10.0

# constants for reward (old)
R_COLLIDE = -10.0       # when the auv collides with an obstacle
R_ARRIVE = 10.0         # when the auv arrives at the target
R_RANGE = 0.1           # this is a scaler to help determine immediate reward at a time step
R_TIME = -0.01          # negative reward (the longer for the auv to reach the goal, the larger this will be)

# constants for reward with habitats
R_OUT_OF_BOUND = -10000
R_CLOSE_TO_BOUND = -10

R_COLLIDE_100 = -10000
R_CLOSE_TO_OBS = -10

R_MAINTAIN_DIST = 5

R_NEW_HAB = 10
R_IN_HAB = 0
R_IN_HAB_TOO_LONG = -10

# determine how quickly the reward will decay when the auv stays in an habitat
# decay rate should be between 0 and 1
R_HAB_DECAY_RATE = 0.01
R_HAB_INIT = 4
R_HAB_OFFSET = 2

# size of the observation space
# the coordinates of the observation space will be based on 
#   the ENV_SIZE and the inital position of auv and the shark
# (unit: m)
ENV_SIZE = 500.0

DEBUG = False
# if PLOT_3D = False, plot the 2d version
PLOT_3D = False

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
        # action: 
        #   a tuple of (v, w), linear velocity and angular velocity
        # range for v (unit: m/s): [-AUV_MAX_V, AUV_MAX_V]
        # range for w (unit: radians): [-AUV_MAX_W, AUV_MAX_W]
        self.action_space = spaces.Box(low = np.array([-AUV_MAX_W]), high = np.array([AUV_MAX_W]), dtype = np.float64)

        self.observation_space = None

        self.auv_init_pos = None
        self.shark_init_pos = None

        # the current state that the env is in
        self.state = None

        self.obstacle_array = []
        self.obstacle_array_for_rendering = []

        self.habitats_array = []
        self.habitats_array_for_rendering = []

        self.live_graph = Live3DGraph(PLOT_3D)

        self.auv_x_array_plot = []
        self.auv_y_array_plot = []
        self.auv_z_array_plot = []

        self.shark_x_array_plot = []
        self.shark_y_array_plot = []
        self.shark_z_array_plot = []

        self.visited_unique_habitat_count = 0


    def init_env(self, auv_init_pos, shark_init_pos, obstacle_array = [], habitat_grid = []):
        """
        Initialize the environment based on the auv and shark's initial position

        Parameters:
            auv_init_pos - an motion plan state object
            shark_init_pos - an motion plan state object
            obstacle_array - an array of motino plan state objects
            habitat_grid - an HabitatGrid object (discretize the environment into grid)

        Return:
            a dictionary with the initial observation of the environment
        """
        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos
        self.obstacle_array_for_rendering = obstacle_array

        self.habitat_grid = habitat_grid
        self.habitats_array_for_rendering = habitat_grid.habitat_array

        self.obstacle_array = []
        for obs in obstacle_array:
            self.obstacle_array.append([obs.x, obs.y, obs.z, obs.size])
        self.obstacle_array = np.array(self.obstacle_array)

        # declare the observation space (required by OpenAI)
        self.observation_space = spaces.Dict({
            'auv_pos': spaces.Box(low = np.array([auv_init_pos.x - ENV_SIZE, auv_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([auv_init_pos.x + ENV_SIZE, auv_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'shark_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'obstacles_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'habitats_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64)
        })

        # initialize the 
        self.init_data_for_plot(auv_init_pos, shark_init_pos)
        
        return self.reset()


    def actions_range(self, N_v, N_w):
        """
        For DQN, discretize the action space of linear velocity and angular velocity based on the bounds

        Parameters:
            N_v - an int, indicating how many options of linear velocity will the auv have
            N_w - an int, indicating how many options of angular velocity will the auv have

        Return:
            a tuple of 2 arrays: (v_options, w_options)
                - v_options: available options for linear velocity
                - w_options: available options for angular velocity
        """

        self.v_options = np.linspace(AUV_MIN_V, AUV_MAX_V, N_v)
        self.w_options = np.linspace(-AUV_MAX_W, AUV_MAX_W, N_w)

        # guaranteed to have 0 as one of the angular velocity
        self.w_options[N_w//2] = 0

        print((self.v_options, self.w_options))
        text = input("stop")

        return (self.v_options, self.w_options)


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

        # calculate the range between the auv and the shark in the previous timestep
        # use this to calculate reward
        """old_range = self.calculate_range(self.state['auv_pos'], self.state['shark_pos'])"""
        
        # get the old position and orientation data for the auv
        x, y, z, theta = self.state['auv_pos']
      
        # for testing, keep track of the distance that the auv has traveled in this time step
        self.distance_traveled = 0

        # calculate the new position and orientation of the auv
        for _ in range(REPEAT_ACTION_TIME):
            theta = angle_wrap(theta + w * DELTA_T)
            dist_x = v * np.cos(theta) * DELTA_T
            x = x + dist_x
            dist_y = v * np.sin(theta) * DELTA_T
            y = y + dist_y
            self.distance_traveled += np.sqrt(dist_x ** 2 + dist_y ** 2)
       
        """shark_x, shark_y, shark_z, shark_theta = self.state['shark_pos']

        # randomly pick the linear and angular velocity of the shark
        shark_v = np.random.uniform(SHARK_MIN_V, SHARK_MAX_V)
        shark_w = np.random.uniform(-SHARK_MAX_W, SHARK_MAX_W)

        for _ in range(REPEAT_ACTION_TIME):
            shark_theta = angle_wrap(shark_theta + shark_w * DELTA_T)
            shark_x = shark_x + shark_v * np.cos(shark_theta) * DELTA_T
            shark_y = shark_y + shark_v * np.sin(shark_theta) * DELTA_T

        new_shark_pos = np.array([shark_x, shark_y, shark_z, shark_theta])"""
        
        # update the current state to the new state
        self.state['auv_pos'] = np.array([x, y, z, theta])

        # calculate the new distance from the wall
        self.state['auv_dist_from_walls'] = self.habitat_grid.distance_from_grid_boundary(self.state['auv_pos'])

        """self.state['shark_pos'] = new_shark_pos"""

        # the episode will only end (done = True) if
        #   - the auv has hit an obstacle, or
        #   - the auv has exceeded the habitat grid
        done = self.check_collision(self.state['auv_pos']) or (not (self.habitat_grid.within_habitat_env(self.state['auv_pos'])))

        # a HabitatCell object, able to indicate which habitat is the auv in currently
        # might be a boolean when the auv is out of bound
        self.visited_habitat_cell = self.habitat_grid.inside_habitat(self.state['auv_pos'])

        # keep track of the total time that the auv has spent in a habitat
        # need to check if it's False, because that indicates that the auv is out of bound
        if self.visited_habitat_cell != False:
            self.total_time_in_hab += 1

        # update the habitats array so that it will increment "the number of time visited" for the habitat that the auv has just visited in this time step
        self.habitats_array = self.update_num_time_visited_for_habitats(self.habitats_array, self.visited_habitat_cell)

        # because we are constantly modifying "number of time visited", 
        # we want to make sure that we are actually storing a new copy of habitats_array into the state, instead of just making a reference
        self.state['habitats_pos'] = copy.deepcopy(self.habitats_array)

        # reward = self.get_reward_with_habitats(self.state['auv_pos'], self.state['shark_pos'], old_range, self.state['habitats_pos'],self.visited_habitat_index_array)
        # reward = self.get_reward_with_habitats_no_decay(self.state['auv_pos'], self.state['shark_pos'], old_range, self.state['habitats_pos'], self.visited_habitat_cell)
        # reward = self.get_range_reward(self.state['auv_pos'], self.state['shark_pos'], old_range)
        reward = self.get_reward_with_habitats_no_shark(self.state['auv_pos'], self.state['habitats_pos'], self.visited_habitat_cell, self.state['auv_dist_from_walls'])

        return self.state, reward, done, {}


    def adjust_action(self, v_init_index, w_init_index, starting_auv_pos, num_of_options_v, num_of_options_w):
        x, y, z, theta = starting_auv_pos

        v_action_index = v_init_index
        w_action_index = w_init_index


        v = self.v_options[v_action_index]
        w = self.w_options[w_action_index]

        # calculate the new position and orientation of the auv
        for _ in range(REPEAT_ACTION_TIME):
            theta = angle_wrap(theta + w * DELTA_T)
            dist_x = v * np.cos(theta) * DELTA_T
            x = x + dist_x
            dist_y = v * np.sin(theta) * DELTA_T
            y = y + dist_y
        
        auv_pos = [x, y, z, theta]

        distance_from_walls = self.habitat_grid.distance_from_grid_boundary(auv_pos)

        if self.check_close_to_walls(auv_pos, distance_from_walls) or self.check_close_to_obstacles(auv_pos):
            # reselect the actions
            v_action_index = num_of_options_v-1
            w_action_index = num_of_options_w-1

        return v_action_index, w_action_index
        
    
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

    
    def within_follow_range(self, auv_pos, shark_pos):
        """
        Check if the auv is within FOLLOWING_RADIUS of the shark

        Parameters:
            auv_pos - an array / a numpy array
            shark_pos - an array / a numpy array
                both have the format: [x_pos, y_pos, z_pos, theta]
        """
        auv_shark_range = self.calculate_range(auv_pos, shark_pos)
        if auv_shark_range <= FOLLOWING_RADIUS:
            if DEBUG:
                print("Within the following range")
            return True
        else:
            return False

    
    def check_collision(self, auv_pos):
        """
        Check if the auv at the current state is hitting any obstacles

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= obs[3]:
                print("Hit an obstacle")
                return True
        return False


    def check_close_to_obstacles(self, auv_pos):
        """
        Check if the auv at the current state is close to any obstacles
        (Within a circular region with radius: obstacle's radius + OBSTACLE_ZONE)

        Parameter:
            auv_pos - an array / a np array [x, y, z, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= (obs[3] + OBSTACLE_ZONE):
                if DEBUG: 
                    print("Close to an obstacles")
                return True
        return False


    def check_close_to_walls(self, auv_pos, dist_from_walls_array):
        for dist_from_wall in dist_from_walls_array:
            if dist_from_wall <= WALL_ZONE:
                if DEBUG:
                    print("Close to the wall")
                return True
        return False
    

    def update_num_time_visited_for_habitats(self, habitats_array, visited_habitat_cell):
        """
        Update the number of times visited for a habitat that has been visited by the auv in the current timestep

        Parameters:
            habitats_array - an array of arrays, where each array represent the state of a habitat
                format: [[hab1_x, hab1_y, hab1_side_length, hab1_num_time_visited], [hab2_x, hab2_y, hab2_side_length, hab2_num_time_visited], ...]
                Warning: this function does not modify habitats_array
            visited_habitat_cell - a HabitatCell object, indicating the current habitat cell that the auv is in
                its id indicates which habitat's num_time_visited should be incremented by 1

        Return:
            a new copy of the habitats_array with the updated number of time visited 
        """
        # print(visited_habitat_cell)

        # make a deep copy of the original habitats array to ensure that we are not modifying the habitats_array that gets passed in
        new_habitats_array = copy.deepcopy(habitats_array)

        # double check that the auv has actually visited a habitat cell and is not outside of the habitat grid
        if visited_habitat_cell != False:
            habitat_index = visited_habitat_cell.habitat_id

            # the 3rd element represents the number of times the AUV has visited an habitat
            new_habitats_array[habitat_index][3] += 1
        
        return new_habitats_array


    def get_range_reward(self, auv_pos, shark_pos, old_range):
        """
        Return the reward that the auv gets at a specific state (at a specific time step)

        The condition is only based on:
            - whether the auv has maintained a FOLLOW_DIST with the shark
            - whether the auv has hit an obstacle
            - the range between the auv and the shark
        """
        if self.within_habitat_env(auv_pos, shark_pos):
            return R_MAINTAIN_DIST
        elif self.check_collision(auv_pos):
            return R_COLLIDE
        else:
            new_range = self.calculate_range(auv_pos, shark_pos)
            # if auv has gotten closer to the shark, will receive positive reward
            #   else, receive negative reward
            range_diff = old_range - new_range
            
            reward = R_RANGE * range_diff
            
            return reward


    def get_reward_with_habitats(self, auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array):
        """
        Return the reward that the auv gets at a specific state (at a specific time step)

        The condition is only based on:
            - whether the auv has hit an obstacle
            - whether the auv is close to an obstacle
            - whether the auv is within a FOLLOW_DIST with the shark
            - whether the auv is visiting any habitat
                - Note: the reward for visiting any habitat will gradually decrease as the auv spends more time in a habitat
            - the range between the auv and the shark
        
        TODO: not updated for habitat grid yet
        """
        # if the auv collides with an obstacle
        if self.check_collision(auv_pos):
            return R_COLLIDE_100
        elif self.check_close_to_obstacles(auv_pos):
            return R_CLOSE_TO_OBS
        # if the auv maintain FOLLOW_DISTANCE with the shark
        elif self.within_follow_range(auv_pos, shark_pos):
            if DEBUG:
                print("following shark")
            reward = R_MAINTAIN_DIST

            # if the auv has visited any habitat in this time step
            for hab_idx in visited_habitat_index_array:
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[3]
                if num_of_time_visited == 1:
                    self.visited_unique_habitat_count += 1
                # reward will decrease as the number of times that auv spent in the habitat increases
                reward += R_HAB_INIT * (1 - R_HAB_DECAY_RATE) ** (num_of_time_visited) - R_HAB_OFFSET
            return reward
        else:
            if DEBUG:
                print("else case in reward")
            new_range = self.calculate_range(auv_pos, shark_pos)
            # if auv has gotten closer to the shark, will receive positive reward
            #   else, receive negative reward
            range_diff = old_range - new_range
            
            reward = R_RANGE * range_diff
            
            return reward


    def get_reward_with_habitats_no_decay(self, auv_pos, shark_pos, old_range, habitats_array, visited_habitat_cell):
        """
        Return the reward that the auv gets at a specific state (at a specific time step)

        The condition is only based on:
            - whether the auv has hit an obstacle
            - whether the auv is close to an obstacle
            - whether the auv is within a FOLLOW_DIST with the shark
            - whether the auv is visiting any new habitat
            - the range between the auv and the shark
        """
        # if the auv collides with an obstacle
        if not self.habitat_grid.within_habitat_env(self.state['auv_pos']):
            return R_OUT_OF_BOUND
        elif self.check_collision(auv_pos):
            return R_COLLIDE_100
        elif self.check_close_to_obstacles(auv_pos):
            return R_CLOSE_TO_OBS
        # if the auv maintain FOLLOW_DISTANCE with the shark
        elif self.within_follow_range(auv_pos, shark_pos):
            reward = R_MAINTAIN_DIST

            if visited_habitat_cell != False:
                # if the auv has visited any habitat in this time step
                hab_idx = visited_habitat_cell.habitat_id
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[3]
                # when the habitat is visited for the first time
                if num_of_time_visited == 1:
                    if DEBUG:
                        print("visit new habitat")
                    self.visited_unique_habitat_count += 1
                    reward += R_NEW_HAB
                elif num_of_time_visited > 1:
                    reward += R_IN_HAB
            
            return reward
        else:
            if DEBUG:
                print("else case in reward")
            new_range = self.calculate_range(auv_pos, shark_pos)
            # if auv has gotten closer to the shark, will receive positive reward
            #   else, receive negative reward
            range_diff = old_range - new_range
            
            reward = R_RANGE * range_diff

            if visited_habitat_cell != False:
                # if the auv has visited any habitat in this time step
                hab_idx = visited_habitat_cell.habitat_id
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[3]
                # when the habitat is visited for the first time
                if num_of_time_visited == 1:
                    if DEBUG:
                        print("visit new habitat")
                    self.visited_unique_habitat_count += 1
                    reward += R_NEW_HAB
                elif num_of_time_visited > 1:
                    reward += R_IN_HAB
            
            return reward

    
    def get_reward_with_habitats_no_shark(self, auv_pos, habitats_array, visited_habitat_cell, dist_from_walls_array = []):
        """
        Return the reward that the auv gets at a specific state (at a specific time step)

        The condition is only based on:
            - whether the auv has hit an obstacle
            - whether the auv is close to an obstacle
            - whether the auv is within a FOLLOW_DIST with the shark
            - whether the auv is visiting any new habitat
            - the range between the auv and the shark
        """
        # if the auv collides with an obstacle
        if not self.habitat_grid.within_habitat_env(self.state['auv_pos']):
            return R_OUT_OF_BOUND
        elif self.check_collision(auv_pos):
            return R_COLLIDE_100
        else:
            if DEBUG:
                print("else case in reward")
            reward = 0

            if visited_habitat_cell != False:
                # if the auv has visited any habitat in this time step
                hab_idx = visited_habitat_cell.habitat_id
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[3]
                # when the habitat is visited for the first time
                if num_of_time_visited == 1:
                    if DEBUG:
                        print("visit new habitat")
                    self.visited_unique_habitat_count += 1
                    reward += R_NEW_HAB
                elif num_of_time_visited >= 50:
                    reward += R_IN_HAB_TOO_LONG
            
            return reward
    

    def reset(self):
        """
        Reset the environment
            - Set the observation to the initial auv and shark position
            - Reset the habitat data (the number of times the auv has visited them)

        Return:
            a dictionary with the initial observation of the environment
        """
        # reset the habitat array, make sure that the number of time visited is cleared to 0
        self.habitats_array = []
        for hab in self.habitats_array_for_rendering:
            self.habitats_array.append([hab.x, hab.y, hab.side_length, hab.num_of_time_visited])
        self.habitats_array = np.array(self.habitats_array)

        # reset the count for how many unique habitat had the auv visited
        self.visited_unique_habitat_count = 0
        # reset the count for how many time steps had the auv visited an habitat
        self.total_time_in_hab = 0

        auv_init_pos = np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta])

        self.state = {
            'auv_pos': auv_init_pos,\
            'obstacles_pos': self.obstacle_array,\
            'habitats_pos': self.habitats_array,\
            'auv_dist_from_walls': self.habitat_grid.distance_from_grid_boundary(auv_init_pos)
        }

        return self.state


    def render(self, mode='human', print_state = True):
        """
        Render the environment by
            - printing out auv and shark's current position
            - returning self.state , so that another helper function can use plot the environment

        Return:
            a dictionary representing the current auv position, shark position, obstacles data, habitats data
        """
        
        """auv_pos = self.state['auv_pos']
        shark_pos = self.state['shark_pos']
        
        if print_state: 
            print("==========================")
            print("auv position: ")
            print("x = ", auv_pos[0], " y = ", auv_pos[1], " z = ", auv_pos[2], " theta = ", auv_pos[3])
            print("shark position: ")
            print("x = ", shark_pos[0], " y = ", shark_pos[1], " z = ", shark_pos[2], " theta = ", shark_pos[3])
            print("==========================")"""

        return self.state


    def render_3D_plot(self, auv_pos, shark_pos):
        """
        Render the environment in a 3D environment

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
            shark_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]

        Warning: 
            - plotting in 3D is very slow
        """
        # add the current position so that we can plot the trajectory line later
        self.auv_x_array_plot.append(auv_pos[0])
        self.auv_y_array_plot.append(auv_pos[1])
        self.auv_z_array_plot.append(auv_pos[2])

        self.shark_x_array_plot.append(shark_pos[0])
        self.shark_y_array_plot.append(shark_pos[1])
        self.shark_z_array_plot.append(shark_pos[2])

        self.live_graph.plot_entity(self.auv_x_array_plot, self.auv_y_array_plot, self.auv_z_array_plot, label = 'auv', color = 'r', marker = ',')

        self.live_graph.plot_entity(self.shark_x_array_plot, self.shark_y_array_plot, self.shark_z_array_plot, label = 'shark', color = 'b', marker = ',')

        # plot the circular region where the auv will be considered following the shark
        goal_region = Circle((shark_pos[0],shark_pos[1]), radius=FOLLOWING_RADIUS, color='b', fill=False)
        self.live_graph.ax.add_patch(goal_region)
        Art3d.pathpatch_2d_to_3d(goal_region, z = shark_pos[2], zdir='z')

        if self.obstacle_array_for_rendering != []:
            self.live_graph.plot_obstacles(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

        for hab in self.habitats_array_for_rendering:
            hab_region = Rectangle((hab.x, hab.y), width=hab.side_length, height=hab.side_length, color='#2a753e', fill=False)
            self.live_graph.ax.add_patch(hab_region)
            Art3d.pathpatch_2d_to_3d(hab_region, z=-10, zdir='z')
        
        self.live_graph.ax.set_xlabel('X')
        self.live_graph.ax.set_ylabel('Y')
        self.live_graph.ax.set_zlabel('Z')

        self.live_graph.ax.legend()
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.0001)

        self.live_graph.ax.clear()


    def render_2D_plot(self, auv_pos, shark_pos=[]):
        """
        Render the environment in a 2D environment

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
            shark_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
        """
        
        self.auv_x_array_plot.append(auv_pos[0])
        self.auv_y_array_plot.append(auv_pos[1])
        self.auv_z_array_plot.append(auv_pos[2])

        """self.shark_x_array_plot.append(shark_pos[0])
        self.shark_y_array_plot.append(shark_pos[1])
        self.shark_z_array_plot.append(shark_pos[2])"""

        auv_ln, = self.live_graph.plot_entity_2D(self.auv_x_array_plot, self.auv_y_array_plot, label = 'auv', color = 'r', marker = ',')

        """shark_ln, = self.live_graph.plot_entity_2D(self.shark_x_array_plot, self.shark_y_array_plot, label = 'shark', color = 'b', marker = ',')"""

        """goal_region = Circle((shark_pos[0],shark_pos[1]), radius=FOLLOWING_RADIUS, color='b', fill=False)
        self.live_graph.ax_2D.add_patch(goal_region)"""

        self.live_graph.ax_2D.legend(["auv"])
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.0001)

        """goal_region.remove()"""
        auv_ln.remove()
        """shark_ln.remove()"""
        
        # self.live_graph.ax_2D.clear()


    def init_live_graph(self, live_graph_2D):
        if live_graph_2D:
            if self.obstacle_array_for_rendering != []:
                self.live_graph.plot_obstacles_2D(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

            for hab in self.habitats_array_for_rendering:
                hab_region = Rectangle((hab.x, hab.y), width = hab.side_length, height = hab.side_length, color='#2a753e', fill=False)
                self.live_graph.ax_2D.add_patch(hab_region)
            
            self.live_graph.ax_2D.set_xlabel('X')
            self.live_graph.ax_2D.set_ylabel('Y')

        

    def init_data_for_plot(self, auv_init_pos, shark_init_pos):
        """
        """
        self.auv_x_array_plot = [auv_init_pos.x]
        self.auv_y_array_plot = [auv_init_pos.y]
        self.auv_z_array_plot = [auv_init_pos.z]

        self.shark_x_array_plot = [shark_init_pos.x]
        self.shark_y_array_plot = [shark_init_pos.y]
        self.shark_z_array_plot = [shark_init_pos.z]
