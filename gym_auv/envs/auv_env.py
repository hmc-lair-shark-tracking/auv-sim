import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as Art3d
import copy

from gym_auv.envs.live3DGraph_auv_env import Live3DGraph

# size of the observation space
# the coordinates of the observation space will be based on 
#   the ENV_SIZE and the inital position of auv and the shark
# (unit: m)
ENV_SIZE = 500.0

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

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 3.0
FOLLOWING_RADIUS = 50.0

# the auv will receive an immediate negative reward if it is close to the obstacles
OBSTACLE_ZONE = 3.0

# constants for reward
R_COLLIDE = -10.0       # when the auv collides with an obstacle
R_ARRIVE = 10.0         # when the auv arrives at the target
R_RANGE = 0.1           # this is a scaler to help determine immediate reward at a time step
R_TIME = -0.01          # negative reward (the longer for the auv to reach the goal, the larger this will be)

# constants for reward with habitats
R_COLLIDE_100 = -10
R_MAINTAIN_DIST = 0.05       
R_IN_HAB = 0.05     
R_NEW_HAB = 0.1 
R_CLOSE_TO_OBS = -1 
# R_IMM_PENALTY = -0.1

REPEAT_ACTION_TIME = 5

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

        self.visited_unique_habitat_count = 0

        self.obstacle_array = []
        self.obstacle_array_for_rendering = []

        self.habitats_array = []
        self.habitats_array_for_rendering = []

        self.live_graph = Live3DGraph(PLOT_3D)

        self.auv_x_array_rl = []
        self.auv_y_array_rl = []
        self.auv_z_array_rl = []

        self.shark_x_array_rl = []
        self.shark_y_array_rl = []
        self.shark_z_array_rl = []


    def init_env(self, auv_init_pos, shark_init_pos, obstacle_array = [], habitats_array = []):
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
        self.habitats_array_for_rendering = habitats_array

        self.obstacle_array = []
        for obs in obstacle_array:
            self.obstacle_array.append([obs.x, obs.y, obs.z, obs.size])
        self.obstacle_array = np.array(self.obstacle_array)

        self.habitats_array = []
        for hab in habitats_array:
            self.habitats_array.append([hab.x, hab.y, hab.z, hab.size, hab.num_of_time_visited])
        self.habitats_array = np.array(self.habitats_array)

        self.observation_space = spaces.Dict({
            'auv_pos': spaces.Box(low = np.array([auv_init_pos.x - ENV_SIZE, auv_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([auv_init_pos.x + ENV_SIZE, auv_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'shark_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'obstacles_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64),
            'habitats_pos': spaces.Box(low = np.array([shark_init_pos.x - ENV_SIZE, shark_init_pos.y - ENV_SIZE, -ENV_SIZE, 0.0]), high = np.array([shark_init_pos.x + ENV_SIZE, shark_init_pos.y + ENV_SIZE, 0.0, 0.0]), dtype = np.float64)
        })

        self.init_data_for_3D_plot(auv_init_pos, shark_init_pos)
        
        return self.reset()


    def actions_range(self, N_v, N_w):
        v_options = np.linspace(AUV_MIN_V, AUV_MAX_V, N_v)
        w_options = np.linspace(-AUV_MAX_W, AUV_MAX_W, N_w)
        # guaranteed to have 0 as one of the angular velocity
        w_options[N_w//2] = 0
        print((v_options, w_options))
        text = input("stop")
        # w_options = [0] * N
        return (v_options, w_options)


    def step(self, action, timestep):
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

        old_range = self.calculate_range(self.state['auv_pos'], self.state['shark_pos'])
        
        # get the old position and orientation data for the auv
        x, y, z, theta = self.state['auv_pos']
      
        self.distance_traveled = 0

        # calculate the new position and orientation of the auv
        for _ in range(REPEAT_ACTION_TIME):
            theta = angle_wrap(theta + w * DELTA_T)
            dist_x = v * np.cos(theta) * DELTA_T
            x = x + dist_x
            dist_y = v * np.sin(theta) * DELTA_T
            y = y + dist_y
            self.distance_traveled += np.sqrt(dist_x ** 2 + dist_y ** 2)
       
        # TODO: For now, the shark's position does not change. Might get updated in the future 
        shark_x, shark_y, shark_z, shark_theta = self.state['shark_pos']

        shark_v = np.random.uniform(SHARK_MIN_V, SHARK_MAX_V)
        shark_w = np.random.uniform(-SHARK_MAX_W, SHARK_MAX_W)
        for _ in range(REPEAT_ACTION_TIME):
            shark_theta = angle_wrap(shark_theta + shark_w * DELTA_T)
            shark_x = shark_x + shark_v * np.cos(shark_theta) * DELTA_T
            shark_y = shark_y + shark_v * np.sin(shark_theta) * DELTA_T

        new_shark_pos = np.array([shark_x, shark_y, shark_z, shark_theta])
        
        # update the current state to the new state
        self.state['auv_pos'] = np.array([x, y, z, theta])
        self.state['shark_pos'] = new_shark_pos

        # the episode will only end (done = True) if
        #   - the auv has reached the target, or
        #   - the auv has hit an obstacle
        done = self.check_collision(self.state['auv_pos'])

        self.visited_habitat_index_array = self.check_in_habitat(self.state['auv_pos'], self.habitats_array)

        self.habitats_array = self.update_num_time_visited_for_habitats(self.habitats_array, self.visited_habitat_index_array)

        self.state['habitats_pos'] = copy.deepcopy(self.habitats_array)

        reward = self.get_reward_with_habitats(self.state['auv_pos'], self.state['shark_pos'], old_range, self.state['habitats_pos'],self.visited_habitat_index_array)
        # reward = self.get_range_reward(self.state['auv_pos'], self.state['shark_pos'], old_range)
        # reward = self.get_range_time_reward(self.state[0], self.state[1], old_range, timestep)
        # reward = self.get_binary_reward(self.state[0], self.state[1])

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
            print("Reached the Goal")
            return True
        else:
            return False

    
    def within_follow_range(self, auv_pos, shark_pos):
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


    def check_close_to_obstacles(self, auv_pos):
        """
        Check if the auv at the current state is hitting any obstacles

        Parameter:
            auv_pos - a np array [x, y, z, theta]
        """
        for obs in self.obstacle_array:
            distance = self.calculate_range(auv_pos, obs)
            # obs[3] indicates the size of the obstacle
            if distance <= (obs[3] + OBSTACLE_ZONE):
                if DEBUG: 
                    print("Close to an obstacles")
                return True
        return False


    def check_in_habitat(self, auv_pos, habitats_array):
        """
        """
        visited_hab_idx_array = []
        for i in range(len(habitats_array)):
            hab = habitats_array[i]
            distance = self.calculate_range(auv_pos, hab)
            if distance <= hab[3]:
                if DEBUG:
                    print("visit habitat #", i)
                visited_hab_idx_array.append(i)
        return visited_hab_idx_array
    

    def update_num_time_visited_for_habitats(self, habitats_array, visited_habitat_index_array):
        """
        """
        new_habitats_array = copy.deepcopy(habitats_array)

        for hab_idx in visited_habitat_index_array:
            new_habitats_array[hab_idx][4] += 1
        
        return new_habitats_array


    def get_range_reward(self, auv_pos, shark_pos, old_range):
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
            
            reward = R_RANGE * range_diff
            
            return reward


    def get_range_time_reward(self, auv_pos, shark_pos, old_range, timestep):
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

            # print("^^^^^^^^^^^^^^^^^^^^^^^")
            # print("old range")
            # print(old_range)
            # print("new range")
            # print(new_range)
            # print("^^^^^^^^^^^^^^^^^^^^^^^")
            
            reward = R_RANGE * range_diff

            # print("just the range reward")
            # print(reward)

            reward += R_TIME * timestep

            # print("with timestep penalty")
            # print(reward)
            
            return reward


    def get_reward_with_habitats(self, auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array):
        # if the auv collides with an obstacle
        if self.check_collision(auv_pos):
            return R_COLLIDE_100
        elif self.check_close_to_obstacles(auv_pos):
            return R_CLOSE_TO_OBS
        # if the auv maintain FOLLOW_DISTANCE with the shark
        elif self.within_follow_range(auv_pos, shark_pos):
            reward = R_MAINTAIN_DIST
            # if the auv has visited any habitat in this time step
            for hab_idx in visited_habitat_index_array:
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[4]
                # when the habitat is visited for the first time
                if num_of_time_visited == 1:
                    if DEBUG:
                        print("visit new habitat")
                    self.visited_unique_habitat_count += 1
                    reward += R_NEW_HAB
                elif num_of_time_visited > 1:
                    reward += R_IN_HAB
            return reward
        elif visited_habitat_index_array != []:
            reward = 0.0
            # if the auv has visited any habitat in this time step
            for hab_idx in visited_habitat_index_array:
                hab = habitats_array[hab_idx]
                num_of_time_visited = hab[4]
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
    

    def reset(self):
        """
        Reset the environment
            Set the observation to the initial auv and shark position
        """
        self.habitats_array = []
        for hab in self.habitats_array_for_rendering:
            self.habitats_array.append([hab.x, hab.y, hab.z, hab.size, hab.num_of_time_visited])
        self.habitats_array = np.array(self.habitats_array)

        # reset the count for how many unique habitat had the auv visited
        self.visited_unique_habitat_count = 0

        self.state = {
            'auv_pos': np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta]),\
            'shark_pos': np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z, self.shark_init_pos.theta]),\
            'obstacles_pos': self.obstacle_array,\
            'habitats_pos': self.habitats_array,\
        }
        return self.state


    def render(self, mode='human', print_state = True):
        """
        Render the environment by
            - printing out auv and shark's current position
            - returning self.state , so that another helper function can use live3DGraph.py class to plot the environment

        Return:
            a tuple of 2 np.array representing the auv and shark's current position
        """
        auv_pos = self.state['auv_pos']
        shark_pos = self.state['shark_pos']
        
        if print_state: 
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

        self.live_graph.plot_entity(self.shark_x_array_rl, self.shark_y_array_rl, self.shark_z_array_rl, label = 'shark', color = 'b', marker = ',')

        goal_region = Circle((shark_pos[0],shark_pos[1]), radius=FOLLOWING_RADIUS, color='b', fill=False)
        self.live_graph.ax.add_patch(goal_region)
        Art3d.pathpatch_2d_to_3d(goal_region, z = shark_pos[2], zdir='z')

        if self.obstacle_array_for_rendering != []:
            self.live_graph.plot_obstacles(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

        for hab in self.habitats_array_for_rendering:
            hab_region = Circle((hab.x, hab.y), radius=hab.size, color='#2a753e', fill=False)
            self.live_graph.ax.add_patch(hab_region)
            Art3d.pathpatch_2d_to_3d(hab_region, z=hab.z, zdir='z')
        
        self.live_graph.ax.set_xlabel('X')
        self.live_graph.ax.set_ylabel('Y')
        self.live_graph.ax.set_zlabel('Z')

        self.live_graph.ax.legend()
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.0001)

        self.live_graph.ax.clear()


    def render_2D_plot(self, auv_pos, shark_pos):
        self.auv_x_array_rl.append(auv_pos[0])
        self.auv_y_array_rl.append(auv_pos[1])
        self.auv_z_array_rl.append(auv_pos[2])

        self.shark_x_array_rl.append(shark_pos[0])
        self.shark_y_array_rl.append(shark_pos[1])
        self.shark_z_array_rl.append(shark_pos[2])

        self.live_graph.plot_entity_2D(self.auv_x_array_rl, self.auv_y_array_rl, label = 'auv', color = 'r', marker = ',')

        self.live_graph.plot_entity_2D(self.shark_x_array_rl, self.shark_y_array_rl, label = 'shark', color = 'b', marker = ',')

        goal_region = Circle((shark_pos[0],shark_pos[1]), radius=FOLLOWING_RADIUS, color='b', fill=False)
        self.live_graph.ax_2D.add_patch(goal_region)

        if self.obstacle_array_for_rendering != []:
            self.live_graph.plot_obstacles_2D(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

        for hab in self.habitats_array_for_rendering:
            hab_region = Circle((hab.x, hab.y), radius=hab.size, color='#2a753e', fill=False)
            self.live_graph.ax_2D.add_patch(hab_region)
        
        self.live_graph.ax_2D.set_xlabel('X')
        self.live_graph.ax_2D.set_ylabel('Y')

        self.live_graph.ax_2D.legend()
        
        plt.draw()

        # pause so the plot can be updated
        plt.pause(0.0001)

        self.live_graph.ax_2D.clear()


    def init_data_for_3D_plot(self, auv_init_pos, shark_init_pos):
        self.auv_x_array_rl = [auv_init_pos.x]
        self.auv_y_array_rl = [auv_init_pos.y]
        self.auv_z_array_rl = [auv_init_pos.z]

        self.shark_x_array_rl = [shark_init_pos.x]
        self.shark_y_array_rl = [shark_init_pos.y]
        self.shark_z_array_rl = [shark_init_pos.z]


    