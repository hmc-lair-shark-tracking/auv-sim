import gym
import math
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T  
import copy

from gym_rrt.envs.motion_plan_state_rrt import Motion_plan_state

from habitatGrid import HabitatGrid

# namedtuple allows us to store Experiences as labeled tuples
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done', 'has_nodes_array'))

"""
============================================================================

    Parameters

============================================================================
"""

# define the range between the starting point of the auv and shark
DIST = 20.0

NUM_OF_EPISODES = 1000
MAX_STEP = 300

NUM_OF_EPISODES_TEST =  50
MAX_STEP_TEST = 300

N_V = 7
N_W = 7

GAMMA = 0.999

EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 0.000075

LEARNING_RATE = 0.001

NEGATIVE_OFFSET = -10000

MEMORY_SIZE = 100000
BATCH_SIZE = 64

# number of additional goals to be added to the replay memory
NUM_GOALS_SAMPLED_HER = 4

TARGET_UPDATE = 10000

NUM_OF_OBSTACLES = 7

ENV_SIZE = 50.0
ENV_GRID_CELL_SIDE_LENGTH = 10.0

# the output size for the neural network
NUM_OF_GRID_CELLS = int((int(ENV_SIZE) / int(ENV_GRID_CELL_SIDE_LENGTH)) ** 2)

# the input size for the neural network
STATE_SIZE = int(8 + NUM_OF_OBSTACLES * 4 + NUM_OF_GRID_CELLS * 3)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many episode should we save the model
SAVE_EVERY = 10
# how many episode should we render the model
RENDER_EVERY = 100
# how many episode should we run a test on the model
TEST_EVERY = 50

FILTER_IN_UPDATING_NN = True

DEBUG = False

RAND_PICK = False
RAND_PICK_RATE = 0.75

R_USEFUL_STATE = 10

"""
============================================================================

    Helper Functions

============================================================================
"""
def process_state_for_nn(state):
    """
    Convert the state (observation in the environment) to a tensor so it can be passed into the neural network

    Parameter:
        state - a direction of two np arrays
    """
    # auv_tensor = torch.from_numpy(state['auv_pos'])
    # shark_tensor = torch.from_numpy(state['shark_pos'])

    # obstacle_tensor = torch.from_numpy(state['obstacles_pos'])
    # obstacle_tensor = torch.flatten(obstacle_tensor)

    rrt_grid_tensor = torch.from_numpy(state['rrt_grid_num_of_nodes_only'])
    # rrt_grid_tensor = torch.flatten(rrt_grid_tensor)

    """habitat_tensor = torch.from_numpy(state['habitats_pos'])
    habitat_tensor = torch.flatten(habitat_tensor)"""
    
    # join tensors together
    return rrt_grid_tensor.float()


def extract_tensors(experiences):
    """
    Convert batches of experiences sampled from the replay memeory to tuples of tensors
    """
    batch = Experience(*zip(*experiences))
   
    t1 = torch.stack(batch.state)
    t2 = torch.stack(batch.action)
    t3 = torch.stack(batch.next_state)
    t4 = torch.cat(batch.reward)
    t5 = torch.stack(batch.done)
    t6 = torch.stack(batch.has_nodes_array)

    return (t1, t2, t3, t4, t5, t6)


def save_model(policy_net, target_net):
    print("Model Save...")
    torch.save(policy_net.state_dict(), 'checkpoint_policy.pth')
    torch.save(target_net.state_dict(), 'checkpoint_target.pth')


def calculate_range(a_pos, b_pos):
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


def validate_new_obstacle(new_obstacle, new_obs_size, auv_init_pos, shark_init_pos, obstacle_array):
    """
    Helper function for checking whether the newly obstacle generated is valid or not
    """
    auv_overlaps = calculate_range([auv_init_pos.x, auv_init_pos.y], new_obstacle) <= new_obs_size
    shark_overlaps = calculate_range([shark_init_pos.x, shark_init_pos.y], new_obstacle) <= new_obs_size
    obs_overlaps = False
    for obs in obstacle_array:
        if calculate_range([obs.x, obs.y], new_obstacle) <= (new_obs_size + obs.size):
            obs_overlaps = True
            break
    return auv_overlaps or shark_overlaps or obs_overlaps


def generate_rand_obstacles(auv_init_pos, shark_init_pos, num_of_obstacles, shark_min_x, shark_max_x,  shark_min_y, shark_max_y):
    """
    """
    obstacle_array = []
    for _ in range(num_of_obstacles):
        obs_x = np.random.uniform(shark_min_x, shark_max_x)
        obs_y = np.random.uniform(shark_min_y, shark_max_y)
        obs_size = np.random.randint(1,5)
        # to prevent this from going into an infinite loop
        counter = 0
        while validate_new_obstacle([obs_x, obs_y], obs_size, auv_init_pos, shark_init_pos, obstacle_array) and counter < 100:
            obs_x = np.random.uniform(shark_min_x, shark_max_x)
            obs_y = np.random.uniform(shark_min_y, shark_max_y)
            counter += 1
        obstacle_array.append(Motion_plan_state(x = obs_x, y = obs_y, z=-5, size = obs_size))

    return obstacle_array   


def validate_new_habitat(new_habitat, new_hab_size, habitats_array):
    """
    Helper function for checking whether the newly habitat generated is valid or not
    """
    hab_overlaps = False
    for hab in habitats_array:
        if calculate_range([hab.x, hab.y], new_habitat) <= (new_hab_size + hab.size):
            hab_overlaps = True
            break
    return hab_overlaps



"""
Class for building policy and target neural network
"""
class Neural_network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_1_in = 600, hidden_layer_1_out = 400, hidden_layer_2_out = 300, hidden_layer_3_out = 200):
        """
        Initialize the Q neural network with input

        Parameter:
            input_size - int, the size of observation space
            output_size_v - int, the number of possible options for v
            output_size_y - int, the number of possible options for w
        """
        super().__init__()
        
        # input layer
        self.input = nn.Linear(in_features = input_size, out_features = hidden_layer_1_in)
        self.bn_in = nn.LayerNorm(hidden_layer_1_in)

        # hidden layer 1, 600 nodes
        self.hidden_1 = nn.Linear(in_features = hidden_layer_1_in, out_features = hidden_layer_1_out) 
        self.bn_h1 = nn.LayerNorm(hidden_layer_1_out)  

        # hidden layer 2, 400 nodes
        self.hidden_2 = nn.Linear(in_features = hidden_layer_1_out, out_features = hidden_layer_2_out)
        self.bn_h2 = nn.LayerNorm(hidden_layer_2_out)

        # hideen layer 3, 300 nodes
        self.hidden_3 = nn.Linear(in_features = hidden_layer_2_out, out_features = hidden_layer_3_out)
        self.bn_h3 = nn.LayerNorm(hidden_layer_3_out)

        self.out = nn.Linear(in_features = hidden_layer_3_out, out_features = output_size)   
        

    def forward(self, t):
        """
        Define the forward pass through the neural network

        Parameters:
            t - the state as a tensor
        """

        # input layer
        t = self.input(t)
        t = F.relu(t)
        t = self.bn_in(t)

        # hidden layer 1
        t = self.hidden_1(t)
        t = F.relu(t)
        t = self.bn_h1(t)

        # hidden layer 2
        t = self.hidden_2(t)
        t = F.relu(t)
        t = self.bn_h2(t)

        # hidden layer 3
        t = self.hidden_3(t)
        t = F.relu(t)
        t = self.bn_h3(t)
  
        t = self.out(t)

        return t



"""
    Class to define replay memeory for training the neural network
"""
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        """
        Store an experience in the replay memory
        Will overwrite any oldest experience first if necessary

        Parameter:
            experience - namedtuples for storing experiences
        """
        # if there's space in the replay memory
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # overwrite the oldest memory
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1


    def sample(self, batch_size):
        """
        Randomly sample "batch_size" amount of experiences from replace memory

        Parameter: 
            batch_size - int, number of experiences that we want to sample from replace memory
        """
        return random.sample(self.memory, batch_size)


    def can_provide_sample(self, batch_size):
        """
        The replay memeory should only sample experiences when it has experiences greater or equal to batch_size

        Parameter: 
            batch_size - int, number of experiences that we want to sample from replace memory
        """
        return len(self.memory) >= batch_size



"""
For implementing epsilon greedy strategy in choosing an action
(exploration vs exploitation)
"""
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        """
        Parameter:
            start - the start value of epsilon
            end - the end value of epsilon
            decay - the decay value of epsilon 
        """
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        """
        Calculate the exploration rate to determine whether the agent should
            explore or exploit in the environment
        """
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)



"""
Class to represent the agent and decide its action in the environment
"""
class Agent():
    def __init__(self, device):
        """
        Parameter: 
            strategy - Epsilon Greedy Strategy class (decide whether we should explore the environment or if we should use the DQN)
            actions_range_v - int, the number of possible values for v that the agent can take
            actions_range_w - int, the number of possible values for w that the agent can take
            device - what we want to PyTorch to use for tensor calculation
        """
        # the agent's current step in the environment
        self.current_step = 0
        
        self.strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
       
        self.device = device

        self.rate = None

        self.neural_net_bad_choice = 0

        self.neural_net_choice = 0

    
    def generate_index_to_pick (self, has_node_array):
        """
        Parameter:
            has_node_array - an array of 0s and 1s, where
                1 indicates that the grid cell is pickable (has at least 1 node in it)
        """
        index_to_pick = []
        for i in range(len(has_node_array)):
            if has_node_array[i] == 1:
                index_to_pick.append(i)

        return index_to_pick


    def select_action(self, state, policy_net, testing = False):
        """
        Pick an action (index to select from array of options for v and from array of options for w)

        Parameters:
            state - tuples for auv position, shark (goal) position, and obstacles position
            policy_net - the neural network to determine the action

        Returns:
            a tensor representing the index for v action and the index for w action
                format: tensor([v_index, w_index])
        """
        if testing:
            # if we are doing intermediate testing
            self.rate = EPS_END
        else:
            self.rate = self.strategy.get_exploration_rate(self.current_step)

            # as the number of steps increases, the exploration rate will decrease
            self.current_step += 1

        index_to_pick = self.generate_index_to_pick(state["has_node"])

        if self.rate > random.random():
            # exploring the environment by randomly chosing an action
            if DEBUG:
                print("-----")
                print("randomly picking")
            
            grid_cell_index = random.choice(index_to_pick)

            return torch.tensor([grid_cell_index]).to(self.device) # explore

        else:
            # turn off gradient tracking bc we are using the model for inference instead of training
            # we don't need to keep track the gradient because we are not doing backpropagation to figure out the weight 
            # of each node yet
            with torch.no_grad():
                # convert the state to a flat tensor to prepare for passing into the neural network
                input_state = process_state_for_nn(state)

                # for the given "state"，the output will be Q values for each possible action (index for a grid cell)
                #   from the policy net
                q_values_all_grid_cells = policy_net(input_state).to(self.device)

                # tensor of 0s and 1s, 1 indicating that a grid cell is valid (has nodes in it)
                has_node_tensor = torch.from_numpy(state["has_node"])

                # filter out the grid cell without any node
                processed_q_values_all_grid_cells = q_values_all_grid_cells+ (1 - has_node_tensor) * NEGATIVE_OFFSET
                
                # pick the grid cell with the largest q value
                grid_cell_index = torch.argmax(processed_q_values_all_grid_cells).item()

                if random.random() < RAND_PICK_RATE and RAND_PICK:
                    grid_cell_index = random.choice(index_to_pick)

                if DEBUG:
                    print("-----")
                    print("exploiting")

                # if state["has_node"][grid_cell_index] == 0:
                #     if DEBUG:
                #         print("has to randomly pick")
                #     self.neural_net_bad_choice += 1
                #     grid_cell_index = random.choice(index_to_pick)

                self.neural_net_choice += 1

                return torch.tensor([grid_cell_index]).to(self.device) # explore  



"""
Class Wrapper for the auv RL environment
"""
class AuvEnvManager():
    def __init__(self, device):
        """
        Parameters: 
            device - what we want to PyTorch to use for tensor calculation
            N - 
            auv_init_pos - 
            shark_init_pos -
            obstacle_array - 
        """
        self.device = device

        # have access to behind-the-scenes dynamics of the environment 
        self.env = gym.make('gym_rrt:rrt-v0').unwrapped

        self.current_state = None
        self.done = False

    
    def init_env_randomly(self, dist = DIST):
        auv_init_pos = Motion_plan_state(x = 10.0, y = 10.0, z = -5.0, theta = 0.0)
        shark_init_pos = Motion_plan_state(x = 35.0, y = 40.0, z = -5.0, theta = 0.0)
        # obstacle_array = generate_rand_obstacles(auv_init_pos, shark_init_pos, NUM_OF_OBSTACLES, shark_min_x, shark_max_x, shark_min_y, shark_max_y)
        obstacle_array = [\
            Motion_plan_state(x=12.0, y=38.0, size=4),\
            Motion_plan_state(x=17.0, y=34.0, size=5),\
            Motion_plan_state(x=20.0, y=29.0, size=4),\
            Motion_plan_state(x=25.0, y=25.0, size=3),\
            Motion_plan_state(x=29.0, y=20.0, size=4),\
            Motion_plan_state(x=34.0, y=17.0, size=3),\
            Motion_plan_state(x=37.0, y=8.0, size=5)\
        ]
        # obstacle_array = [\
        #     Motion_plan_state(x=12.0, y=38.0, size=4),\
        #     Motion_plan_state(x=17.0, y=34.0, size=5),\
        #     Motion_plan_state(x=20.0, y=29.0, size=4),\
        #     Motion_plan_state(x=25.0, y=25.0, size=3),\
        #     Motion_plan_state(x=29.0, y=20.0, size=4),\
        #     Motion_plan_state(x=34.0, y=17.0, size=3),\
        # ]

        boundary_array = [Motion_plan_state(x=0.0, y=0.0), Motion_plan_state(x = ENV_SIZE, y = ENV_SIZE)]

        # self.habitat_grid = HabitatGrid(habitat_bound_x, habitat_bound_y, habitat_bound_size_x, habitat_bound_size_y, HABITAT_SIDE_LENGTH, HABITAT_CELL_SIDE_LENGTH)

        print("===============================")
        print("Starting Positions")
        print(auv_init_pos)
        print(shark_init_pos)
        print("-")
        print(obstacle_array)
        print("-")
        print("Number of Environment Grid")
        print(NUM_OF_GRID_CELLS)
        print("===============================")

        if DEBUG:
            text = input("stop")

        return self.env.init_env(auv_init_pos, shark_init_pos, boundary_array = boundary_array, grid_cell_side_length = ENV_GRID_CELL_SIDE_LENGTH, obstacle_array = obstacle_array)


    def reset(self):
        """
        Reset the environment and return the initial state
        """
        return self.env.reset()


    def close(self):
        self.env.close()


    def render(self, mode='human', print_state = True, live_graph_2D = False):
        """
        Render the environment both as text in terminal and as a 3D graph if necessary

        Parameter:
            mode - string, modes for rendering, currently only supporting "human"
            live_graph - boolean, will display the 3D live_graph if True
        """
        if live_graph_2D:
            self.env.render_2D_plot(self.current_state["path"])
            

    def reset_render_graph(self, live_graph_2D = False):
        if live_graph_2D:
            self.env.live_graph.ax_2D.clear()


    def take_action(self, chosen_grid_cell_index, step_num):
        """
        Parameter: 
            action - tensor of the format: tensor([v_index, w_index])
                use the index from the action and take a step in environment
                based on the chosen values for v and w
            step_num - to allow us to identify the useful states
        """
        chosen_grid_cell_index = chosen_grid_cell_index.item()
        # we only care about the reward and whether or not the episode has ended
        # action is a tensor, so item() returns the value of a tensor (which is just a number)
        self.current_state, reward, self.done, _ = self.env.step(chosen_grid_cell_index, step_num)

        if DEBUG:
            print("=========================")
            print("step num")
            print(step_num)
            print("chosen grid cell index: ")
            print(chosen_grid_cell_index)
            print("chosen grid: ")
            print(self.current_state["rrt_grid"][chosen_grid_cell_index])
            print("------")
            print("new state: ")
            print(self.current_state)
            print("reward: ")
            print(reward)
            print("=========================")

        # wrap reward into a tensor, so we have input and output to both be tensor
        return torch.tensor([reward], device=self.device).float()

    def get_state(self):
        """
        state will be represented as the difference bewteen 2 screens
            so we can calculate the velocity
        """
        return self.env.state

    
    def get_range_reward(self, auv_pos, goal_pos, old_range):
        reward = self.env.get_range_reward(auv_pos, goal_pos, old_range)

        return torch.tensor([reward], device=self.device).float()

    
    def get_range_time_reward(self, auv_pos, goal_pos, old_range, timestep):
        reward = self.env.get_range_time_reward(auv_pos, goal_pos, old_range, timestep)

        return torch.tensor([reward], device=self.device).float()


    def get_binary_reward(self, auv_pos, goal_pos):
        """
        Wrapper to convert the binary reward (-1 or 1) to a tensor

        Parameters:
            auv_pos - an array of the form [x, y, z, theta]
            goal_pos - an array of the same form, represent the target position that the auv is currently trying to reach
        """
        reward = self.env.get_binary_reward(auv_pos, goal_pos)

        return torch.tensor([reward], device=self.device).float()


    def get_reward_with_habitats(self, auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array):

        reward = self.env.get_reward_with_habitats(auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array)

        return torch.tensor([reward], device=self.device).float()


    def get_reward_with_habitats_no_decay(self, auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array):

        reward = self.env.get_reward_with_habitats_no_decay(auv_pos, shark_pos, old_range, habitats_array, visited_habitat_index_array)

        return torch.tensor([reward], device=self.device).float()


    def get_reward_with_habitats_no_shark(self, auv_pos, habitats_array, visited_habitat_cell, dist_from_walls_array = []):
        reward = self.env.get_reward_with_habitats_no_shark(auv_pos, habitats_array, visited_habitat_cell, dist_from_walls_array)

        return torch.tensor([reward], device=self.device).float()


class QValues():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # actions is a tensor with this format: [[v_action_index1, w_action_index1], [v_action_index2, w_action_index2] ]
        # actions[:,:1] gets only the first element in the [v_action_index, w_action_index], 
        #   so we get all the v_action_index as a tensor
        # policy_net(states) gives all the predicted q-values for all the action outcome for a given state
        # policy_net(states).gather(dim=1, index=actions[:,:1]) gives us
        #   a tensor of the q-value corresponds to the state and action(specified by index=actions[:,:1]) pair 
        
        q_values = policy_net(states).gather(dim=1, index=actions[:,:1])

        return q_values

    
    @staticmethod        
    def get_next(target_net, next_states, has_nodes_arrays):  
        # for each next state, we want to obtain the max q-value predicted by the target_net among all the possible next actions              
        # we want to know where the final states are bc we shouldn't pass them into the target net  
        if FILTER_IN_UPDATING_NN:
            q_values_for_all_actions = target_net(next_states)

            # remove the q values of the invalid grid cells
            processed_q_values_for_all_actions = q_values_for_all_actions + (1 - has_nodes_arrays) * NEGATIVE_OFFSET

            processed_max_q_values = processed_q_values_for_all_actions.max(dim=1)[0].detach()

            return processed_max_q_values
        else:
            return target_net(next_states).max(dim=1)[0].detach()           


class DQN():
    def __init__(self):
        # initialize the policy network and the target network
        self.policy_net = Neural_network(NUM_OF_GRID_CELLS, NUM_OF_GRID_CELLS).to(DEVICE)
        self.target_net = Neural_network(NUM_OF_GRID_CELLS, NUM_OF_GRID_CELLS).to(DEVICE)

        self.hard_update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.policy_net_optim = optim.Adam(params = self.policy_net.parameters(), lr = LEARNING_RATE)

        self.memory = ReplayMemory(MEMORY_SIZE)

        # set up the environment
        self.em = AuvEnvManager(DEVICE)

        self.agent = Agent(DEVICE)


    def hard_update(self, target, source):
        """
        Make sure that the target have the same parameters as the source
            Used to initialize the target networks for the actor and critic
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def load_trained_network(self):
        """
        Load already trained neural network
        """
        print("Loading previously trained neural network...")
        self.agent.strategy.start = EPS_END
        self.policy_net.load_state_dict(torch.load('checkpoint_policy.pth'))
        self.target_net.load_state_dict(torch.load('checkpoint_target.pth'))


    def plot_summary_graph (self, episode_array, upper_plot_y_data, upper_plot_ylabel, upper_plot_title, lower_plot_y_data, lower_plot_ylabel, lower_plot_title):
        # close plot if there is any
        plt.close()

        fig = plt.figure(figsize= [10, 8])

        ax_upper = fig.add_subplot(2, 1, 1)
        ax_lower = fig.add_subplot(2, 1, 2)

        ax_upper.plot(episode_array, upper_plot_y_data)
        ax_lower.plot(episode_array, lower_plot_y_data)

        ax_upper.scatter(episode_array, upper_plot_y_data, color='b')
        ax_lower.scatter(episode_array, lower_plot_y_data, color='b')

        ax_upper.set_title(upper_plot_title)
        ax_lower.set_title(lower_plot_title)

        ax_upper.set_ylabel(upper_plot_ylabel)

        ax_lower.set_xlabel("number of episodes trained")
        ax_lower.set_ylabel(lower_plot_ylabel)

        plt.show()
    

    def plot_intermediate_testing_result(self, starting_distance, episode_array, result_array):
        """
        Plot the following graphs
            1.  average total reward vs episodes
                average duration in episodes vs episodes
            2.  success rate vs epsiodes
                success rate (normalized) vs  episodes
            3.  number of unique habitats visited vs episodes
                number of unique habitats visited (normalized) vs episodes
            4.  average timestep spent in the habitat vs episodes
                average timestep spent in the habitat (normalized) vs episodes
        """
        # for plot #1 
        avg_total_reward_array = []
        avg_episode_duration_array = []

        # for plot #2
        success_rate_array = []
        success_rate_array_norm = []

        # for plot #3
        bad_choices_array = []
        bad_choices_over_total_choices_array = []

        # generate array of data so it's easy to plot
        for result in result_array:
            # for plot #1 
            avg_total_reward_array.append(result["avg_total_reward"])
            avg_episode_duration_array.append(result["avg_eps_duration"])

            # for plot #2
            success_rate_array.append(result["success_rate"])
            success_rate_array_norm.append(result["success_rate_norm"])

            # for plot #3
            bad_choices_array.append(result["bad_choices"])
            bad_choices_over_total_choices_array.append(result["bad_choices_over_total_choices"])

        
        # begin plotting the graph
        # plot #1: 
        #   average total reward vs episodes
        #   average duration in episodes vs episodes
        upper_plot_title = "average total reward vs. episodes"
        upper_plot_ylabel = "average total reward"

        lower_plot_title = "average episode duration vs episodes"
        lower_plot_ylabel = "avg episode duration (steps)"

        self.plot_summary_graph(episode_array, avg_total_reward_array, upper_plot_ylabel, upper_plot_title, \
            avg_episode_duration_array, lower_plot_ylabel, lower_plot_title)

        # plot #2: 
        #   success rate vs epsiodes
        #   success rate (normalized) vs  episodes
        upper_plot_title = "success rate vs. episodes"
        upper_plot_ylabel = "success rate (%)"

        lower_plot_title = "success rate (divided by avg episode duraiton) vs. episodes"
        lower_plot_ylabel = "success rate (%)"

        self.plot_summary_graph(episode_array, success_rate_array, upper_plot_ylabel, upper_plot_title, \
            success_rate_array_norm, lower_plot_ylabel, lower_plot_title)

        # plot #2: 
        #   success rate vs epsiodes
        #   success rate (normalized) vs  episodes
        upper_plot_title = "avg number of bad choices by neural net vs. episodes"
        upper_plot_ylabel = "bad choices"

        lower_plot_title = "avg number of bad choices over total choices by neural net vs. episodes"
        lower_plot_ylabel = "bad choices / total choices"

        self.plot_summary_graph(episode_array, bad_choices_array, upper_plot_ylabel, upper_plot_title, \
            bad_choices_over_total_choices_array, lower_plot_ylabel, lower_plot_title)

        
        # print out the result so that we can save for later
        print("episode tested")
        print(episode_array)
        text = input("stop")

        # for plot #1 
        print("total reward array")
        print(avg_total_reward_array)
        print("average episode duration")
        print(avg_episode_duration_array)
        text = input("stop")

        # for plot #2
        print("success")
        print(success_rate_array)
        print(success_rate_array_norm)
        text = input("stop")

        # for plot #2
        print("choice")
        print(bad_choices_array)
        print(bad_choices_over_total_choices_array)
        text = input("stop")


    def extract_useful_states(self, path):
        """
        Parameter:
            path - might be 1. list of motion plan state, representing the final path
                            2. [BAD] a motion plan state, representing a node added to the tree
                                (so the RRT planner was not able to find a valid path)
                            3. [BAD] None, representing that there isn't any new node added to the tree
                                (so the RRT planner was not able to find a valid path)
        """
        useful_state_idx_array = []

        # path is a valid parameter, the rrt planner actually found a valid path from start to goal
        if type(path) == list:
            # add the first step into the useful states first
            useful_state_idx_array.append(path[0].rl_state_id)
            
            for i in range(1, len(path)):
                pt = path[i]
                # prevent adding repeated state id or start and goal
                if pt.rl_state_id != useful_state_idx_array[-1] and pt.rl_state_id != None:
                    useful_state_idx_array.append(pt.rl_state_id)

            # remove the first useful state because that state already receives a large final reward
            useful_state_idx_array = useful_state_idx_array[1:]

        return useful_state_idx_array


    def post_process_reward_array_from_path(self, reward_array, useful_state_idx_array):
        """
        Modify the reward by boosting the reward for useful states

        Parameters:
            reward_array - stores the reward for all the states
            useful_state_idx_array - store the step number of the useful states
                (states that contribute to creating the final path)

        Warning:
            This function modifies the reward_array directly
        """
        for idx in useful_state_idx_array:
            # we have to do idx-1 because the first for loop starts out at 1
            # we have to subtract by 1 to get the right index to modify the reward
            reward_array[idx-1] =  torch.tensor([R_USEFUL_STATE], device=DEVICE).float()
        
        return reward_array


    def save_real_experiece(self, state, next_state, action, reward, done):
        """old_range = calculate_range(state['auv_pos'], state['shark_pos'])"""

        """visited_habitat_cell = self.em.habitat_grid.inside_habitat(next_state['auv_pos'])"""

        """reward = self.em.get_reward_with_habitats_no_decay(next_state['auv_pos'], next_state['shark_pos'], old_range,\
            next_state['habitats_pos'], visited_habitat_cell)"""
        
        """reward = self.em.get_reward_with_habitats_no_shark(next_state['auv_pos'], next_state['habitats_pos'], visited_habitat_cell, next_state['auv_dist_from_walls'])"""

        self.memory.push(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward, done, torch.from_numpy(next_state["has_node"])))

        # print("**********************")
        # print("real experience")
        # print(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward, done))
        # text = input("stop")

    
    def generate_extra_goals(self, time_step, next_state_array):
        additional_goals = []

        possible_goals_to_sample = next_state_array[time_step+1: ]

        # only sample additional goals if there are enough to sample
        # TODO: slightly modified from our previous implementation of HER, maybe this is better?
        if len(possible_goals_to_sample) >= NUM_GOALS_SAMPLED_HER:
            additional_goals = random.sample(possible_goals_to_sample, k = NUM_GOALS_SAMPLED_HER)
        
        return additional_goals

    
    def store_extra_goals_HER(self, state, next_state, action, additional_goals, timestep):
        for goal in additional_goals:
            new_curr_state = {\
                'auv_pos': state['auv_pos'],\
                'obstacles_pos': state['obstacles_pos'],\
                'habitats_pos': state['habitats_pos'],\
                'auv_dist_from_walls': state['auv_dist_from_walls']
            }

            visited_habitat_cell = self.em.habitat_grid.inside_habitat(next_state['auv_pos'])

            new_habitats_array = self.em.env.update_num_time_visited_for_habitats(new_curr_state['habitats_pos'], visited_habitat_cell)

            new_next_state = {\
                'auv_pos': next_state['auv_pos'],\
                'obstacles_pos': next_state['obstacles_pos'],\
                'habitats_pos': new_habitats_array,\
                'auv_dist_from_walls': next_state['auv_dist_from_walls']
            }
            
            """old_range = calculate_range(new_curr_state['auv_pos'], new_curr_state['shark_pos'])

            reward = self.em.get_reward_with_habitats_no_decay(new_next_state['auv_pos'], new_next_state['shark_pos'], old_range,\
                new_next_state['habitats_pos'], visited_habitat_cell)"""

            reward = self.em.get_reward_with_habitats_no_shark(new_next_state['auv_pos'], new_next_state['habitats_pos'], visited_habitat_cell, new_next_state['auv_dist_from_walls'])

            done = torch.tensor([0], device=DEVICE).int()
            if self.em.env.check_collision(new_next_state['auv_pos']) or (not (self.em.habitat_grid.within_habitat_env(new_next_state['auv_pos']))):
                done = torch.tensor([1], device=DEVICE).int()
                
            self.memory.push(Experience(process_state_for_nn(new_curr_state), action, process_state_for_nn(new_next_state), reward, done))

            # print("------------")
            # print("HER experience")
            # print(Experience(process_state_for_nn(new_curr_state), action, process_state_for_nn(new_next_state), reward, done))
            # text = input("stop")
    

    def update_neural_net(self):
        if self.memory.can_provide_sample(BATCH_SIZE):
            # Sample random batch from replay memory.
            experiences = self.memory.sample(BATCH_SIZE)
            
            # extract states, actions, rewards, next_states into their own individual tensors from experiences batch
            states, actions, next_states, rewards, dones, has_nodes_arrays = extract_tensors(experiences)

            # Pass a batch of states and actions to policy network.
            # return the q value for the given state-action pair
            current_q_values = QValues.get_current(self.policy_net, states, actions)

            # Pass a batch of next_states to the target network
            # get the maximum predicted Q values among all actions for each next_state
            next_q_values = QValues.get_next(self.target_net, next_states, has_nodes_arrays)

            target_q_values = (next_q_values * GAMMA * (1 - dones.flatten())) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

            self.loss_in_eps.append(loss.item())

            self.policy_net_optim.zero_grad()
            
            loss.backward()

            # gradient clipping
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.policy_net_optim.step()


    def train(self, num_episodes, max_step, load_prev_training = False, use_HER = True, live_graph_2D = False):
        episode_durations = []
        avg_loss_in_training = []
        total_reward_in_training = []

        episodes_that_got_tested = []
        testing_result_array = []

        num_of_bad_choices = []
        num_of_bad_choices_over_total_choices = []

        target_update_counter = 1

        if load_prev_training:
            # if we want to continue training an already trained network
            self.load_trained_network()
        
        for eps in range(1, num_episodes+1):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly()

            # reward received in this episode
            eps_reward = 0

            action_array = []
            next_state_array = []
            reward_array = []
            done_array = []

            # determine how many steps we should run HER
            # by default, it will be "max_step" - 1 because in the first loop, we start at t=1
            iteration = max_step - 1

            self.loss_in_eps = []

            self.agent.neural_net_bad_choice = 0
            self.agent.neural_net_choice = 0

            if (eps % RENDER_EVERY == 0) and live_graph_2D:
                self.em.env.init_live_graph(live_graph_2D = live_graph_2D)

            for t in range(1, max_step):
                chosen_grid_cell_index = self.agent.select_action(state, self.policy_net)

                action_array.append(chosen_grid_cell_index)

                score = self.em.take_action(chosen_grid_cell_index, t)
                eps_reward += score.item()
                reward_array.append(score)

                next_state = copy.deepcopy(self.em.get_state())
                next_state_array.append(next_state)

                done_array.append(torch.tensor([0], device=DEVICE).int())

                if (eps % RENDER_EVERY == 0) and live_graph_2D:
                    self.em.render(print_state = False, live_graph_2D = live_graph_2D)
                    
                    # if DEBUG:
                    #     text = input("stop")

                state = next_state

                if self.em.done:
                    iteration = t
                    done_array[t-1] = torch.tensor([1], device=DEVICE).int()
                    break
            
            num_of_bad_choices.append(self.agent.neural_net_bad_choice)
            num_of_bad_choices_over_total_choices.append(self.agent.neural_net_choice)
            episode_durations.append(iteration)
            total_reward_in_training.append(eps_reward)
            
            useful_state_idx_array = self.extract_useful_states(state["path"])

            reward_array = self.post_process_reward_array_from_path(reward_array, useful_state_idx_array)
            
            # reset the state before we start updating the neural network
            state = self.em.reset()

            # reset the rendering
            if (eps % RENDER_EVERY == 0) and live_graph_2D:
                self.em.reset_render_graph(live_graph_2D = live_graph_2D)

            for t in range(iteration):
                action = action_array[t]
                next_state = next_state_array[t]
                done = done_array[t]
                reward = reward_array[t]

                # store the actual experience that the auv has in the first loop into the memory
                self.save_real_experiece(state, next_state, action, reward, done)

                if use_HER:
                    additional_goals = self.generate_extra_goals(t, next_state_array)
                    self.store_extra_goals_HER(state, next_state, action, additional_goals, t)

                state = next_state

                self.update_neural_net()
                
                target_update_counter += 1

                if target_update_counter % TARGET_UPDATE == 0:
                    print("UPDATE TARGET NETWORK")
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # print("*********************************")
            # print("final state")
            # print(state)

            if self.loss_in_eps != []:
                avg_loss = np.mean(self.loss_in_eps)
                avg_loss_in_training.append(avg_loss)
                print("+++++++++++++++++++++++++++++")
                print("Episode # ", eps, "end with reward: ", score, "total reward: ", eps_reward, "average loss", avg_loss, " used time: ", iteration)
                print("+++++++++++++++++++++++++++++")
            else:
                print("+++++++++++++++++++++++++++++")
                print("Episode # ", eps, "end with reward: ", score, "total reward: ", eps_reward, "average loss nan", " used time: ", iteration)
                print("+++++++++++++++++++++++++++++")

            # if eps % TARGET_UPDATE == 0:
            #     print("UPDATE TARGET NETWORK")
            #     self.target_net.load_state_dict(self.policy_net.state_dict())

            if eps % SAVE_EVERY == 0:
                save_model(self.policy_net, self.target_net)

            if eps % TEST_EVERY == 0:
                episodes_that_got_tested.append(eps)

                result = self.test_model_during_training(NUM_OF_EPISODES_TEST, MAX_STEP_TEST, DIST)

                testing_result_array.append(result)

                
        save_model(self.policy_net, self.target_net)

        self.em.close()

        self.plot_intermediate_testing_result(DIST, episodes_that_got_tested, testing_result_array)

        print("exploration rate")
        print(self.agent.rate)
        text = input("stop")

        print("episode duration")
        print(episode_durations)
        text = input("stop")

        print("average loss")
        print(avg_loss_in_training)
        text = input("stop")

        print("total reward in training")
        print(total_reward_in_training)

    
    def test(self, num_episodes, max_step, live_graph_2D = False):
        # modify the starting distance betweeen the auv and the shark to prepare for testing
        episode_durations = []
        total_reward_array = []

        bad_choices_array = []
        bad_choices_over_total_choices_array = []

        success_count = 0

        self.load_trained_network()
        self.policy_net.eval()
        
        for eps in range(num_episodes):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly()

            episode_durations.append(max_step)

            eps_reward = 0.0

            self.agent.neural_net_bad_choice = 0
            self.agent.neural_net_choice = 0

            if live_graph_2D:
                self.em.env.init_live_graph(live_graph_2D = live_graph_2D)

            for t in range(1, max_step):
                chosen_grid_cell_index = self.agent.select_action(state, self.policy_net)

                reward = self.em.take_action(chosen_grid_cell_index, t)

                eps_reward += reward.item()

                self.em.render(print_state = False, live_graph_2D = live_graph_2D)

                state = self.em.get_state()

                if self.em.done:
                    # because the only way for an episode to terminate is when an rrt path is found
                    success_count += 1
                    episode_durations[eps] = t
                    break
            
            if live_graph_2D:
                self.em.reset_render_graph(live_graph_2D = live_graph_2D)
            
            print("+++++++++++++++++++++++++++++")
            print("Test Episode # ", eps, "end with reward: ", eps_reward, " used time: ", episode_durations[-1])
            print("+++++++++++++++++++++++++++++")

            total_reward_array.append(eps_reward)

        self.em.close()

        print("final sums of time")
        print(episode_durations)
        print("average time")
        print(np.mean(episode_durations))
        print("-----------------")

        text = input("stop")

        print("total reward")
        print(total_reward_array)
        print("average total reward")
        print(np.mean(total_reward_array))
        print("-----------------")

        text = input("stop")

        print("success count")
        print(success_count)



    def test_model_during_training (self, num_episodes, max_step, starting_dist):
        # modify the starting distance betweeen the auv and the shark to prepare for testing
        episode_durations = []
        total_reward_array = []

        bad_choices_array = []
        bad_choices_over_total_choices_array = []

        success_count = 0

        # assuming that we are testing the model during training, so we don't need to load the model 
        self.policy_net.eval()
        
        for eps in range(num_episodes):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly(starting_dist)

            episode_durations.append(max_step)

            eps_reward = 0.0

            self.agent.neural_net_bad_choice = 0
            self.agent.neural_net_choice = 0

            for t in range(1, max_step):
                chosen_grid_cell_index = self.agent.select_action(state, self.policy_net, testing = True)

                reward = self.em.take_action(chosen_grid_cell_index, t)

                eps_reward += reward.item()

                state = self.em.get_state()

                if self.em.done:
                    # because the only way for an episode to terminate is when an rrt path is found
                    success_count += 1
                    episode_durations[eps] = t
                    break

            bad_choices_array.append(self.agent.neural_net_bad_choice)

            if self.agent.neural_net_choice != 0:
                bad_choices_over_total_choices_array.append(float(self.agent.neural_net_bad_choice) / float(self.agent.neural_net_choice))
            else:
                bad_choices_over_total_choices_array.append(0.0)
               
            print("+++++++++++++++++++++++++++++")
            print("Test Episode # ", eps, "end with reward: ", eps_reward, " used time: ", episode_durations[-1])
            print("+++++++++++++++++++++++++++++")

            total_reward_array.append(eps_reward)

        self.policy_net.train()

        avg_total_reward = np.mean(total_reward_array)

        avg_episode_duration = np.mean(episode_durations)

        success_rate = float(success_count) / float(num_episodes) * 100

        # we can normalize it by the average of traveled distance
        success_rate_normalized = float(success_rate) / avg_episode_duration

        result = {
            "avg_total_reward": avg_total_reward,
            "avg_eps_duration": avg_episode_duration,
            "success_rate": success_rate,
            "success_rate_norm": success_rate_normalized,
            "bad_choices": np.mean(bad_choices_array),
            "bad_choices_over_total_choices": np.mean(bad_choices_over_total_choices_array)
        }

        return result
    

def main():
    dqn = DQN()
   
    dqn.train(NUM_OF_EPISODES, MAX_STEP, load_prev_training = False, live_graph_2D = True, use_HER = False)
    # dqn.test(NUM_OF_EPISODES_TEST, MAX_STEP_TEST, live_graph_2D = False)
    # dqn.test_q_value_control_auv(NUM_OF_EPISODES_TEST, MAX_STEP_TEST, live_graph_3D = False, live_graph_2D = True)


if __name__ == "__main__":
    main()