import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import copy
import random
import torch

# Warning: Comment out matplotlib library if we are using XSEDE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
import mpl_toolkits.mplot3d.art3d as Art3d
from gym_rrt.envs.live3DGraph_rrt_env import Live3DGraph

from gym_rrt.envs.rrt_dubins import Planner_RRT
from gym_rrt.envs.motion_plan_state_rrt import Motion_plan_state

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

RRT_PLANNER_FREQ = 10

# the maximum range between the auv and shark to be considered that the auv has reached the shark
END_GAME_RADIUS = 3.0
FOLLOWING_RADIUS = 50.0

# the auv will receive an immediate negative reward if it is close to the obstacles
OBSTACLE_ZONE = 0.0
WALL_ZONE = 10.0

# constants for reward
R_FOUND_PATH = 300
R_CREATE_NODE = 0
R_INVALID_NODE = -1

REMOVE_CELL_WITH_MANY_NODES = False

# size of the observation space
# the coordinates of the observation space will be based on 
#   the ENV_SIZE and the inital position of auv and the shark
# (unit: m)
ENV_SIZE = 500.0

DEBUG = False
# if PLOT_3D = False, plot the 2d version
PLOT_3D = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

"""
============================================================================

    Helper Functions

============================================================================
"""
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


"""
============================================================================

    Class - RRT Env

============================================================================
"""
class RRTEnv(gym.Env):
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


    def init_env(self, auv_init_pos, shark_init_pos, boundary_array, grid_cell_side_length, num_of_subsections, obstacle_array = [], empty_slot_tensor = [], habitat_grid = None):
        """
        Initialize the environment based on the auv and shark's initial position

        Parameters:
            auv_init_pos - an motion plan state object
            shark_init_pos - an motion plan state object
            boundary_array - an array of 2 motion plan state objects
                TODO: For now, let the environment be a rectangle
                1st mps represents the bottom left corner of the env
                2nd mps represents the upper right corner of the env
            obstacle_array - an array of motion plan state objects
            habitat_grid - an HabitatGrid object (discretize the environment into grid)

        Return:
            a dictionary with the initial observation of the environment
        """
        self.auv_init_pos = auv_init_pos
        self.shark_init_pos = shark_init_pos

        self.obstacle_array_for_rendering = obstacle_array
        
        self.habitats_array_for_rendering = []
        if habitat_grid != None:
            self.habitat_grid = habitat_grid
            self.habitats_array_for_rendering = habitat_grid.habitat_array

        self.obstacle_array = []
        for obs in obstacle_array:
            self.obstacle_array.append([obs.x, obs.y, obs.z, obs.size])
        self.obstacle_array = np.array(self.obstacle_array)

        self.empty_slot_tensor = empty_slot_tensor

        self.boundary_array = boundary_array

        self.cell_side_length = grid_cell_side_length
        self.num_of_subsections = num_of_subsections

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

    def step(self, chosen_grid_cell_idx, step_num):
        """
        In each step, we will generate an additional node in the RRT tree.

        Parameter:
            action - a tuple, representing the linear velocity and angular velocity of the auv

        Return:
            observation - a tuple of 2 np array, representing the auv and shark's new position
                each array has the format: [x_pos, y_pos, z_pos, theta]
            reward - float, amount of reward returned after previous action
            done - float, whether the episode has ended
            info - dictionary, can provide debugging info (TODO: right now, it's just an empty one)
        """
        # print("picked unprocessed index")
        # print(chosen_grid_cell_idx)

        # print("state")
        # for i in range(len(self.state["rrt_grid"])):
        #     print(str(i) + " : " + str(self.state["rrt_grid"][i]))

        grid_cell_index = chosen_grid_cell_idx // self.num_of_subsections
        subsection_index = chosen_grid_cell_idx % self.num_of_subsections
        
        # print("grid cell index")
        # print(grid_cell_index)

        # print("subsection index")
        # print(subsection_index)

        # convert the index for grid cells in the 1D array back to 2D array
        chosen_grid_cell_row_idx = grid_cell_index // len(self.rrt_planner.env_grid[0])
        chosen_grid_cell_col_idx = grid_cell_index % len(self.rrt_planner.env_grid[0])

        # print("row and col")
        # print(chosen_grid_cell_row_idx)
        # print(chosen_grid_cell_col_idx)

        # text = input("stop")

        # chosen_grid_cell = self.rrt_planner.env_grid[chosen_grid_cell_row_idx][chosen_grid_cell_col_idx].subsection_cells[subsection_index]

        done, path = self.rrt_planner.generate_one_node((chosen_grid_cell_row_idx, chosen_grid_cell_col_idx, subsection_index), step_num, remove_cell_with_many_nodes=REMOVE_CELL_WITH_MANY_NODES)

        # TODO: how we are updating the grid's info and the has node array is very inefficient

        self.state["has_node"] = np.array(self.rrt_planner.has_node_array)

        self.state["rrt_grid_num_of_nodes_only"] = np.array(self.rrt_planner.rrt_grid_1D_array_num_of_nodes_only)

        if path != None:
            self.state["path"] = path

        # if the RRT planner has found a path in this step
        if done and path != None:
            reward = R_FOUND_PATH
        elif path != None:
            # if the RRT planner adds a new node
            reward = R_CREATE_NODE
        else:
            # TODO: For now, the reward encourages using less time to plan the path
            reward = R_INVALID_NODE
        
        
        return self.state, reward, done, {}


    def convert_rrt_grid_to_1D (self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """
        rrt_grid_1D_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    rrt_grid_1D_array.append([grid_cell.x, grid_cell.y, subsection.theta, len(subsection.node_array)])

        return np.array(rrt_grid_1D_array)


    def convert_rrt_grid_to_1D_num_of_nodes_only(self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """
        rrt_grid_1D_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    rrt_grid_1D_array.append(len(subsection.node_array))

        return np.array(rrt_grid_1D_array)

    def generate_rrt_grid_has_node_array (self, rrt_grid):
        """
        Parameter:
            rrt_grid - a 2D array, represent all the grid cells
        """

        has_node_array = []

        for row in rrt_grid:
            for grid_cell in row:
                for subsection in grid_cell.subsection_cells:
                    if len(subsection.node_array) == 0:
                        has_node_array.append(0)
                    else:
                        has_node_array.append(1)

        return np.array(has_node_array)

    
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
    

    def reset(self):
        """
        Reset the environment
            - Set the observation to the initial auv and shark position
            - Reset the habitat data (the number of times the auv has visited them)

        Return:
            a dictionary with the initial observation of the environment
        """
        # reset the habitat array, make sure that the number of time visited is cleared to 0
        # self.habitats_array = []
        # for hab in self.habitats_array_for_rendering:
        #     self.habitats_array.append([hab.x, hab.y, hab.side_length, hab.num_of_time_visited])
        # self.habitats_array = np.array(self.habitats_array)

        # reset the count for how many unique habitat had the auv visited
        self.visited_unique_habitat_count = 0
        # reset the count for how many time steps had the auv visited an habitat
        self.total_time_in_hab = 0

        auv_init_pos = np.array([self.auv_init_pos.x, self.auv_init_pos.y, self.auv_init_pos.z, self.auv_init_pos.theta])

        shark_init_pos = np.array([self.shark_init_pos.x, self.shark_init_pos.y, self.shark_init_pos.z, self.shark_init_pos.theta])

        # initialize the RRT planner
        self.rrt_planner = Planner_RRT(self.auv_init_pos, self.shark_init_pos, self.boundary_array, self.obstacle_array_for_rendering, self.habitats_array_for_rendering, cell_side_length = self.cell_side_length, dist_to_end = 1, diff_max = 0.75, freq=20, subsections_in_cell = self.num_of_subsections)

        # rrt_grid_1D_array = self.convert_rrt_grid_to_1D(self.rrt_planner.env_grid)
        # rrt_grid_1D_array_num_of_nodes_only = self.convert_rrt_grid_to_1D_num_of_nodes_only(self.rrt_planner.env_grid)
        # has_node_array = self.generate_rrt_grid_has_node_array(self.rrt_planner.env_grid)

        self.state = {
            'auv_pos': auv_init_pos,\
            'shark_pos': torch.from_numpy(shark_init_pos).float().to(DEVICE),\
            'obstacles_pos': self.obstacle_array,\
            'has_node': np.array(self.rrt_planner.has_node_array),\
            'path': None,\
            'rrt_grid_num_of_nodes_only': np.array(self.rrt_planner.rrt_grid_1D_array_num_of_nodes_only),\
            'empty_slot_tensor': self.empty_slot_tensor,\
        }

        # print("initial state")
        # print(has_node_array)
        # print(self.state["has_node"])
        # print(has_node_array == self.state["has_node"])
        # print("---")
        # print(rrt_grid_1D_array_num_of_nodes_only)
        # print(self.state["rrt_grid_num_of_nodes_only"])
        # print(rrt_grid_1D_array_num_of_nodes_only == self.state["rrt_grid_num_of_nodes_only"])
        # text = input("stop")

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


    def render_2D_plot(self, new_state):
        """
        Render the environment in a 2D environment

        Parameters:
            auv_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
            shark_pos - an array / a numpy array, with format [x_pos, y_pos, z_pos, theta]
        """
        if new_state != None and type(new_state) != list:
            # draw the new edge, which is not a successful path
            self.live_graph.ax_2D.plot([point.x for point in new_state.path], [point.y for point in new_state.path], '-', color="#000000")
            self.live_graph.ax_2D.plot(new_state.x, new_state.y, 'o', color="#000000")
        elif new_state != None and type(new_state) == list:
            # if we are supposed to draw the final path  
            # new_state is now a list of nodes
            self.live_graph.ax_2D.plot([node.x for node in new_state], [node.y for node in new_state], '-r')
            # self.ax.plot(rnd.x, rnd.y, ",", color="#000000")

        # pause so the plot can be updated
        plt.pause(0.0001)


    def init_live_graph(self, live_graph_2D):
        if live_graph_2D:
            self.live_graph.ax_2D.plot(self.auv_init_pos.x, self.auv_init_pos.y, "xr")
            self.live_graph.ax_2D.plot(self.shark_init_pos.x, self.shark_init_pos.y, "xr")

            if self.obstacle_array_for_rendering != []:
                self.live_graph.plot_obstacles_2D(self.obstacle_array_for_rendering, OBSTACLE_ZONE)

            for row in self.rrt_planner.env_grid:
                for grid_cell in row:
                    cell = Rectangle((grid_cell.x, grid_cell.y), width=grid_cell.side_length, height=grid_cell.side_length, color='#2a753e', fill=False)
                    self.live_graph.ax_2D.add_patch(cell)
            
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
