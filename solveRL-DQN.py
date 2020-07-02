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

from motion_plan_state import Motion_plan_state

from habitatGrid import HabitatGrid

# namedtuple allows us to store Experiences as labeled tuples
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

"""
============================================================================

    Parameters

============================================================================
"""

# define the range between the starting point of the auv and shark
DIST = 20.0

NUM_OF_EPISODES = 500
MAX_STEP = 1000

NUM_OF_EPISODES_TEST =  1000
MAX_STEP_TEST = 1000

N_V = 7
N_W = 7

GAMMA = 0.999

EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 0.001

LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 64

# number of additional goals to be added to the replay memory
NUM_GOALS_SAMPLED_HER = 4

TARGET_UPDATE = 10000

NUM_OF_OBSTACLES = 4

HABITAT_SIDE_LENGTH = 20
HABITAT_CELL_SIDE_LENGTH = 20
NUM_OF_HABITATS = int((DIST * 20 / HABITAT_SIDE_LENGTH) ** 2)

STATE_SIZE = 4 + NUM_OF_OBSTACLES * 4 + NUM_OF_HABITATS * 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how many episode should we save the model
SAVE_EVERY = 10
# how many episode should we render the model
RENDER_EVERY = 250
# how many episode should we run a test on the model
TEST_EVERY = 100

DEBUG = False

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
    auv_tensor = torch.from_numpy(state['auv_pos'])
    """shark_tensor = torch.from_numpy(state['shark_pos'])"""

    obstacle_tensor = torch.from_numpy(state['obstacles_pos'])
    obstacle_tensor = torch.flatten(obstacle_tensor)

    habitat_tensor = torch.from_numpy(state['habitats_pos'])
    habitat_tensor = torch.flatten(habitat_tensor)
    
    # join tensors together
    return torch.cat((auv_tensor, obstacle_tensor, habitat_tensor)).float()


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

    return (t1, t2, t3, t4, t5)


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


# def generate_rand_habitats(num_of_habitats, habitat_bound_min_x, habitat_bound_max_x,  habitat_bound_min_y, habitat_bound_max_y):
#     """
#     """
#     habitats_array = []
#     for _ in range(num_of_habitats):
#         hab_x = np.random.uniform(habitat_bound_min_x, habitat_bound_max_x)
#         hab_y = np.random.uniform(habitat_bound_min_y, habitat_bound_max_y)
#         hab_size = np.random.randint(4,11)
#         # to prevent this from going into an infinite loop
#         counter = 0
#         while validate_new_habitat([hab_x, hab_y], hab_size, habitats_array) and counter < 100:
#             hab_x = np.random.uniform(habitat_bound_min_x, habitat_bound_max_x)
#             hab_y = np.random.uniform(habitat_bound_min_y, habitat_bound_max_y)
#             counter += 1
#         habitats_array.append(HabitatState(x = hab_x, y = hab_y, z=-10, size = hab_size))

#     return habitats_array  


"""
Class for building policy and target neural network
"""
class Neural_network(nn.Module):
    def __init__(self, input_size, output_size_v, output_size_w, hidden_layer_in = 400, hidden_layer_out = 300):
        """
        Initialize the Q neural network with input

        Parameter:
            input_size - int, the size of observation space
            output_size_v - int, the number of possible options for v
            output_size_y - int, the number of possible options for w
        """
        super().__init__()
        
        # self.bn0 = nn.LayerNorm(input_size)

        self.fc1 = nn.Linear(in_features = input_size, out_features = hidden_layer_in)
        self.bn1 = nn.LayerNorm(hidden_layer_in)

        # branch for selecting v
        self.fc2_v = nn.Linear(in_features = hidden_layer_in, out_features = hidden_layer_out) 
        self.bn2_v = nn.LayerNorm(hidden_layer_out)     
        self.out_v = nn.Linear(in_features = hidden_layer_out, out_features = output_size_v)
     
        # branch for selecting w
        self.fc2_w = nn.Linear(in_features = hidden_layer_in, out_features = hidden_layer_out)
        self.bn2_w = nn.LayerNorm(hidden_layer_out)     
        self.out_w = nn.Linear(in_features = hidden_layer_out, out_features = output_size_w)
        

    def forward(self, t):
        """
        Define the forward pass through the neural network

        Parameters:
            t - the state as a tensor
        """
        # pass through the layers then have relu applied to it
        # relu is the activation function that will turn any negative value to 0,
        #   and keep any positive value
        # t = self.bn0(t)

        t = self.fc1(t)
        t = F.relu(t)
        t = self.bn1(t)

        # the neural network is separated into 2 separate branch
        t_v = self.fc2_v(t)
        t_v = F.relu(t_v)
        t_v = self.bn2_v(t_v)

        t_w = self.fc2_w(t)
        t_w = F.relu(t_w)
        t_w = self.bn2_w(t_w)
  
        # pass through the last layer, the output layer
        # output is a tensor of Q-Values for all the optinons for v/w
        t_v = self.out_v(t_v)  
        t_w = self.out_w(t_w)

        return torch.stack((t_v, t_w))



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
    def __init__(self, actions_range_v, actions_range_w, device):
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
        
        self.actions_range_v = actions_range_v
        self.actions_range_w = actions_range_w
       
        self.device = device

        self.rate = None


    def select_action(self, state, policy_net):
        """
        Pick an action (index to select from array of options for v and from array of options for w)

        Parameters:
            state - tuples for auv position, shark (goal) position, and obstacles position
            policy_net - the neural network to determine the action

        Returns:
            a tensor representing the index for v action and the index for w action
                format: tensor([v_index, w_index])
        """
        self.rate = self.strategy.get_exploration_rate(self.current_step)
        # as the number of steps increases, the exploration rate will decrease
        self.current_step += 1

        if self.rate > random.random():
            # exploring the environment by randomly chosing an action
            if DEBUG:
                print("-----")
                print("randomly picking")
            v_action_index = random.choice(range(self.actions_range_v))
            w_action_index = random.choice(range(self.actions_range_w))

            return torch.tensor([v_action_index, w_action_index]).to(self.device) # explore

        else:
            # turn off gradient tracking bc we are using the model for inference instead of training
            # we don't need to keep track the gradient because we are not doing backpropagation to figure out the weight 
            # of each node yet
            with torch.no_grad():
                # convert the state to a flat tensor to prepare for passing into the neural network
                state = process_state_for_nn(state)

                # for the given "state"，the output will be Q values for each possible action (index for v and w)
                #   from the policy net
                output_weight = policy_net(state).to(self.device)
                if DEBUG:
                    print("-----")
                    print("exploiting")
                    print("Q values check - v")
                    print(output_weight[0])
                    print("Q values check - w")
                    print(output_weight[1])

                # output_weight[0] is for the v_index, output_weight[1] is for w_index
                # this is finding the index with the highest Q value
                v_action_index = torch.argmax(output_weight[0]).item()
                w_action_index = torch.argmax(output_weight[1]).item()

                return torch.tensor([v_action_index, w_action_index]).to(self.device) # explore  



"""
Class Wrapper for the auv RL environment
"""
class AuvEnvManager():
    def __init__(self, N_v, N_w, device):
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
        self.env = gym.make('gym_auv:auv-v0').unwrapped

        self.current_state = None
        self.done = False

        # an array of the form:
        #   [[array of options for v], [array of options for w]]
        # values of v and w for the agent to chose from
        self.possible_actions = self.env.actions_range(N_v, N_w)

    
    def init_env_randomly(self, dist = DIST):
        auv_min_x = dist * 9
        auv_max_x = dist * 11
        auv_min_y = dist * 9
        auv_max_y = dist * 11

        shark_min_x = dist * 8
        shark_max_x = dist * 12
        shark_min_y = dist * 8
        shark_max_y = dist * 12

        habitat_bound_x = 0.0
        habitat_bound_size_x = dist * 20
        habitat_bound_y = 0.0
        habitat_bound_size_y = dist * 20

        auv_init_pos = Motion_plan_state(x = np.random.uniform(auv_min_x, auv_max_x), y = np.random.uniform(auv_min_y, auv_max_y), z = -5.0, theta = 0)
        shark_init_pos = Motion_plan_state(x = np.random.uniform(shark_min_x, shark_max_x), y = np.random.uniform(shark_min_y, shark_max_y), z = -5.0, theta = np.random.uniform(-np.pi, np.pi))
        # obstacle_array = generate_rand_obstacles(auv_init_pos, shark_init_pos, NUM_OF_OBSTACLES, shark_min_x, shark_max_x, shark_min_y, shark_max_y)
        obstacle_array = [\
            Motion_plan_state(x = shark_min_x, y = shark_min_y, z = -5.0, size = 5),
            Motion_plan_state(x = shark_min_x, y = shark_max_y, z = -5.0, size = 5),
            Motion_plan_state(x = shark_max_x, y = shark_min_y, z = -5.0, size = 5),
            Motion_plan_state(x = shark_max_x, y = shark_max_y, z = -5.0, size = 5),
        ]

        self.habitat_grid = HabitatGrid(habitat_bound_x, habitat_bound_y, habitat_bound_size_x, habitat_bound_size_y, HABITAT_SIDE_LENGTH, HABITAT_CELL_SIDE_LENGTH)

        print("===============================")
        print("Starting Positions")
        print(auv_init_pos)
        print(shark_init_pos)
        print("-")
        print(obstacle_array)
        print("Number of habitats")
        print(NUM_OF_HABITATS)
        print(len(self.habitat_grid.habitat_array))
        print("===============================")

        if DEBUG:
            text = input("stop")

        return self.env.init_env(auv_init_pos, shark_init_pos, obstacle_array, self.habitat_grid)


    def reset(self):
        """
        Reset the environment and return the initial state
        """
        return self.env.reset()


    def close(self):
        self.env.close()


    def render(self, mode='human', print_state = True, live_graph_3D = False, live_graph_2D = False):
        """
        Render the environment both as text in terminal and as a 3D graph if necessary

        Parameter:
            mode - string, modes for rendering, currently only supporting "human"
            live_graph - boolean, will display the 3D live_graph if True
        """
        state = self.env.render(mode, print_state)
        if live_graph_3D: 
            self.env.render_3D_plot(state['auv_pos'], state['shark_pos'])

        if live_graph_2D:
            """self.env.render_2D_plot(state['auv_pos'], state['shark_pos'])"""
            self.env.render_2D_plot(state['auv_pos'])
            
        return state

    def reset_render_graph(self, mode='human', live_graph_3D = False, live_graph_2D = False):
        if live_graph_3D:
            self.env.live_graph.ax.clear()
        elif live_graph_2D:
            self.env.live_graph.ax_2D.clear()


    def take_action(self, action):
        """
        Parameter: 
            action - tensor of the format: tensor([v_index, w_index])
                use the index from the action and take a step in environment
                based on the chosen values for v and w
        """
        v_action_index = action[0].item()
        w_action_index = action[1].item()
        v_action = self.possible_actions[0][v_action_index]
        w_action = self.possible_actions[1][w_action_index]
        
        # we only care about the reward and whether or not the episode has ended
        # action is a tensor, so item() returns the value of a tensor (which is just a number)
        self.current_state, reward, self.done, _ = self.env.step((v_action, w_action))

        if DEBUG:
            print("=========================")
            print("action v: ", v_action_index, " | ", v_action)  
            print("action w: ", w_action_index, " | ", w_action)  
            print("new state: ")
            print(self.current_state)
            print("reward: ")
            print(reward)
            print("=========================")
            text = input("stop")

        # wrap reward into a tensor, so we have input and output to both be tensor
        return torch.tensor([reward], device=self.device).float()

    
    def adjust_action(self, action, state, num_of_options_v, num_of_options_w):
        v_action_index = action[0].item()
        w_action_index = action[1].item()
        
        new_v_action_index, new_w_action_index = self.env.adjust_action(v_action_index, w_action_index, state["auv_pos"], num_of_options_v, num_of_options_w)

        return torch.tensor([new_v_action_index, new_w_action_index]).to(self.device) # explore  


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


    def get_reward_with_habitats_no_shark(self, auv_pos, habitats_array, visited_habitat_cell):
        reward = self.env.get_reward_with_habitats_no_shark(auv_pos, habitats_array, visited_habitat_cell)

        return torch.tensor([reward], device=self.device).float()

"""
Use QValues class's 
"""
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
        
        q_values_for_v = policy_net(states)[0].gather(dim=1, index=actions[:,:1])
        q_values_for_w = policy_net(states)[1].gather(dim=1, index=actions[:,1:2])
       
        return torch.stack((q_values_for_v, q_values_for_w), dim = 0)

    
    @staticmethod        
    def get_next(target_net, next_states):  
        # for each next state, we want to obtain the max q-value predicted by the target_net among all the possible next actions              
        # we want to know where the final states are bc we shouldn't pass them into the target net
       
        v_max_q_values = target_net(next_states)[0].max(dim=1)[0].detach()
        w_max_q_values = target_net(next_states)[1].max(dim=1)[0].detach()
       
        return torch.stack((v_max_q_values, w_max_q_values), dim = 0)



class DQN():
    def __init__(self, N_v, N_w):
        # initialize the policy network and the target network
        self.policy_net = Neural_network(STATE_SIZE, N_v, N_w).to(DEVICE)
        self.target_net = Neural_network(STATE_SIZE, N_v, N_w).to(DEVICE)

        self.hard_update(self.target_net, self.policy_net)
        self.target_net.eval()

        self.policy_net_optim = optim.Adam(params = self.policy_net.parameters(), lr = LEARNING_RATE)

        self.memory = ReplayMemory(MEMORY_SIZE)

        # set up the environment
        self.em = AuvEnvManager(N_v, N_w, DEVICE)

        self.agent = Agent(N_v, N_w, DEVICE)


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
        self.agent.strategy.start = EPS_DECAY + 0.001
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
                average distance between auv and the shark vs episodes
            2.  collision rate vs epsiodes
                collision rate (normalized) vs  episodes
            3.  number of unique habitats visited vs episodes
                number of unique habitats visited (normalized) vs episodes
            4.  average timestep spent in the habitat vs episodes
                average timestep spent in the habitat (normalized) vs episodes
        """
        # for plot #1 
        avg_total_reward_array = []
        avg_dist_btw_auv_shark_array = []

        # for plot #2
        collision_rate_array = []
        collision_rate_array_norm = []

        # for plot #3
        avg_unique_hab_visited_array = []
        avg_unique_hab_visited_array_norm = []

        # for plot #4
        avg_timestep_in_hab_array = []
        avg_timestep_in_hab_array_norm = []

        # generate array of data so it's easy to plot
        for result in result_array:
            # for plot #1 
            avg_total_reward_array.append(result["avg_total_reward"])
            avg_dist_btw_auv_shark_array.append(0)
            # for plot #2
            collision_rate_array.append(result["collision_rate"])
            collision_rate_array_norm.append(result["collision_rate_norm"])

            # for plot #3
            avg_unique_hab_visited_array.append(result["avg_unique_hab_visited"])
            avg_unique_hab_visited_array_norm.append(result["avg_unique_hab_visited_norm"])

            # for plot #4
            avg_timestep_in_hab_array.append(result["avg_timestep_in_hab"])
            avg_timestep_in_hab_array_norm.append(result["avg_timestep_in_hab_norm"])
        
        # begin plotting the graph
        # plot #1: 
        #   average total reward vs episodes
        #   average distance between auv and the shark vs episodes
        upper_plot_title = "average total reward vs. episodes at range = " + str(starting_distance) + 'm'
        upper_plot_ylabel = "average total reward"

        lower_plot_title = "average distance between auv and the shark vs. episodes at range = " + str(starting_distance) + 'm'
        lower_plot_ylabel = "avg dist btw the auv and shark (m)"

        self.plot_summary_graph(episode_array, avg_total_reward_array, upper_plot_ylabel, upper_plot_title, \
            avg_dist_btw_auv_shark_array, lower_plot_ylabel, lower_plot_title)


        # plot #2: 
        #   collision rate vs epsiodes
        #   collision rate (normalized) vs  episodes
        upper_plot_title = "collision rate vs. episodes at range = " + str(starting_distance) + 'm'
        upper_plot_ylabel = "collision rate (%)"

        lower_plot_title = "collision rate (divided by traveled dist) vs. episodes at range = " + str(starting_distance) + 'm'
        lower_plot_ylabel = "collision rate (%)"

        self.plot_summary_graph(episode_array, collision_rate_array, upper_plot_ylabel, upper_plot_title, \
            collision_rate_array_norm, lower_plot_ylabel, lower_plot_title)


        # plot #3: 
        #   number of unique habitats visited vs episodes
        #   number of unique habitats visited (normalized) vs episodes
        upper_plot_title = "avg num of visited unique habitats vs. episodes at range = " + str(starting_distance) + 'm'
        upper_plot_ylabel = "unique visited habitats"

        lower_plot_title = "avg num of visited unique habitats (divided by traveled dist) vs. episodes at range = " + str(starting_distance) + 'm'
        lower_plot_ylabel = "unique visited habitats"

        self.plot_summary_graph(episode_array, avg_unique_hab_visited_array, upper_plot_ylabel, upper_plot_title, \
            avg_unique_hab_visited_array_norm, lower_plot_ylabel, lower_plot_title)


        # plot #4: 
        #   average timestep spent in the habitat vs episodes
        #   average timestep spent in the habitat (normalized) vs episodes
        upper_plot_title = "avg timesteps in habitat vs. episodes at range = " + str(starting_distance) + 'm'
        upper_plot_ylabel = "timesteps"

        lower_plot_title = "avg timesteps in habitat (divided by traveled dist) vs. episodes at range = " + str(starting_distance) + 'm'
        lower_plot_ylabel = "timesteps"

        self.plot_summary_graph(episode_array, avg_timestep_in_hab_array, upper_plot_ylabel, upper_plot_title, \
            avg_timestep_in_hab_array_norm, lower_plot_ylabel, lower_plot_title)
        
        # print out the result so that we can save for later
        print("episode tested")
        print(episode_array)
        text = input("stop")

        # for plot #1 
        print("total reward array")
        print(avg_total_reward_array)
        print("average distance between auv and shark array")
        print(avg_dist_btw_auv_shark_array)
        text = input("stop")

        # for plot #2
        print("collision")
        print(collision_rate_array)
        print(collision_rate_array_norm)
        text = input("stop")

        # for plot #3
        print(" avg_unique_hab_visited_array")
        print(avg_unique_hab_visited_array)
        print(avg_unique_hab_visited_array_norm)
        text = input("stop")

        # for plot #4
        print("avg_timestep_in_hab_array")
        print(avg_timestep_in_hab_array)
        print(avg_timestep_in_hab_array_norm)
        text = input("stop")


    def save_real_experiece(self, state, next_state, action, done, timestep):
        """old_range = calculate_range(state['auv_pos'], state['shark_pos'])"""

        visited_habitat_cell = self.em.habitat_grid.inside_habitat(next_state['auv_pos'])

        """reward = self.em.get_reward_with_habitats_no_decay(next_state['auv_pos'], next_state['shark_pos'], old_range,\
            next_state['habitats_pos'], visited_habitat_cell)"""
        reward = self.em.get_reward_with_habitats_no_shark(next_state['auv_pos'], next_state['habitats_pos'], visited_habitat_cell)

        self.memory.push(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward, done))

        # print("**********************")
        # # print(next_state['habitats_pos'])
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
                'habitats_pos': state['habitats_pos']\
            }

            visited_habitat_cell = self.em.habitat_grid.inside_habitat(next_state['auv_pos'])

            new_habitats_array = self.em.env.update_num_time_visited_for_habitats(new_curr_state['habitats_pos'], visited_habitat_cell)

            new_next_state = {\
                'auv_pos': next_state['auv_pos'],\
                'obstacles_pos': next_state['obstacles_pos'],\
                'habitats_pos': new_habitats_array\
            }
            
            """old_range = calculate_range(new_curr_state['auv_pos'], new_curr_state['shark_pos'])

            reward = self.em.get_reward_with_habitats_no_decay(new_next_state['auv_pos'], new_next_state['shark_pos'], old_range,\
                new_next_state['habitats_pos'], visited_habitat_cell)"""

            reward = self.em.get_reward_with_habitats_no_shark(new_next_state['auv_pos'], new_next_state['habitats_pos'], visited_habitat_cell)

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
            states, actions, next_states, rewards, dones = extract_tensors(experiences)

            # Pass batch of preprocessed states to policy network.
            # return the q value for the given state-action pair by passing throught the policy net
            current_q_values = QValues.get_current(self.policy_net, states, actions)
        
            next_q_values = QValues.get_next(self.target_net, next_states)

            target_q_values_v = (next_q_values[0] * GAMMA * (1 - dones.flatten())) + rewards

            target_q_values_w = (next_q_values[1] * GAMMA * (1 - dones.flatten())) + rewards

            loss_v = F.mse_loss(current_q_values[0], target_q_values_v.unsqueeze(1))
            loss_w = F.mse_loss(current_q_values[1], target_q_values_w.unsqueeze(1))

            loss_total = loss_v + loss_w
            self.loss_in_eps.append(loss_total.item())

            self.policy_net_optim.zero_grad()
            loss_total.backward()
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.policy_net_optim.step()


    def train(self, num_episodes, max_step, load_prev_training = False, use_HER = True, live_graph_3D = False, live_graph_2D = False):
        episode_durations = []
        avg_loss_in_training = []
        total_reward_in_training = []

        episodes_that_got_tested = []
        testing_result_array = []

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
            done_array = []

            # determine how many steps we should run HER
            # by default, it will be "max_step" - 1 because in the first loop, we start at t=1
            iteration = max_step - 1

            self.loss_in_eps = []

            if (eps % RENDER_EVERY == 0) and (live_graph_2D or live_graph_3D):
                self.em.env.init_live_graph(live_graph_2D = live_graph_2D)

            for t in range(1, max_step):
                action = self.agent.select_action(state, self.policy_net)

                # print("old action")
                # print(action)

                # action = self.em.adjust_action(action, state, self.agent.actions_range_v, self.agent.actions_range_w)
                # print(action)
                # print("new action")

                action_array.append(action)

                score = self.em.take_action(action)
                eps_reward += score.item()

                next_state = copy.deepcopy(self.em.get_state())
                next_state_array.append(next_state)

                done_array.append(torch.tensor([0], device=DEVICE).int())

                if (eps % RENDER_EVERY == 0) and (live_graph_2D or live_graph_3D):
                    self.em.render(print_state = False, live_graph_3D = live_graph_3D, live_graph_2D = live_graph_2D)

                state = next_state

                if self.em.done:
                    iteration = t
                    done_array[t-1] = torch.tensor([1], device=DEVICE).int()
                    break
            
            episode_durations.append(iteration)

            total_reward_in_training.append(eps_reward)

            # reset the state before we start updating the neural network
            state = self.em.reset()

            # reset the rendering
            if (eps % RENDER_EVERY == 0) and (live_graph_2D or live_graph_3D):
                self.em.reset_render_graph(live_graph_3D = live_graph_3D, live_graph_2D = live_graph_2D)

            for t in range(iteration):
                action = action_array[t]
                next_state = next_state_array[t]
                done = done_array[t]
                
                # store the actual experience that the auv has in the first loop into the memory
                self.save_real_experiece(state, next_state, action, done, t)

                if use_HER:
                    additional_goals = self.generate_extra_goals(t, next_state_array)
                    self.store_extra_goals_HER(state, next_state, action, additional_goals, t)

                state = next_state

                self.update_neural_net()
                
                target_update_counter += 1

                if target_update_counter % TARGET_UPDATE == 0:
                    print("UPDATE TARGET NETWORK")
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            print("*********************************")
            print("final state")
            print(state)

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

    
    def test(self, num_episodes, max_step, live_graph_3D = False, live_graph_2D = False):
        episode_durations = []
        starting_dist_array = []
        traveled_dist_array = []
        final_reward_array = []
        total_reward_array = []

        # if we want to continue training an already trained network
        self.load_trained_network()
        self.policy_net.eval()
        
        for eps in range(num_episodes):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly()

            """starting_dist = calculate_range(state['auv_pos'], state['shark_pos'])
            starting_dist_array.append(starting_dist)"""

            episode_durations.append(max_step)
            traveled_dist_array.append(0.0)
            final_reward_array.append(0.0)
            total_reward_array.append(0.0)

            reward = 0

            if (live_graph_2D or live_graph_3D):
                self.em.env.init_live_graph(live_graph_2D = live_graph_2D)

            for t in range(1, max_step):
                action = self.agent.select_action(state, self.policy_net)

                reward = self.em.take_action(action)

                final_reward_array[eps] = reward.item()
                total_reward_array[eps] += reward.item()

                traveled_dist_array[eps] += self.em.env.distance_traveled

                self.em.render(print_state = False, live_graph_3D = live_graph_3D, live_graph_2D = live_graph_2D)

                state = self.em.get_state()

                if self.em.done:
                    episode_durations[eps] = t
                    break

            self.em.reset_render_graph(live_graph_3D = live_graph_3D, live_graph_2D = live_graph_2D)
            
            print("+++++++++++++++++++++++++++++")
            print("Episode # ", eps, "end with reward: ", reward, " used time: ", episode_durations[-1])
            print("+++++++++++++++++++++++++++++")


        self.em.close()

        print("final sums of time")
        print(episode_durations)
        print("average time")
        print(np.mean(episode_durations))
        print("-----------------")

        text = input("stop")

        """print("all the starting distances")
        print(starting_dist_array)
        print("average starting distance")
        print(np.mean(starting_dist_array))
        print("-----------------")

        text = input("stop")"""

        print("all the traveled distances")
        print(traveled_dist_array)
        print("average traveled dissta")
        print(np.mean(traveled_dist_array))
        print("-----------------")

        text = input("stop")

        print("final reward")
        print(final_reward_array)
        print("-----------------")

        text = input("stop")

        print("total reward")
        print(total_reward_array)
        print("-----------------")


    def test_model_during_training (self, num_episodes, max_step, starting_dist):
        # modify the starting distance betweeen the auv and the shark to prepare for testing
        episode_durations = []
        total_reward_array = []
        traveled_dist_array = []

        time_in_habitat_array = []
        num_unique_habitat_array = []

        time_in_habitat_array_normalized = []
        num_unique_habitat_array_normalized = []

        collision_count = 0

        # assuming that we are testing the model during training, so we don't need to load the model 
        self.policy_net.eval()
        
        for eps in range(num_episodes):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly(starting_dist)
            
            # store the distance between the auv and the shark at each step
            dist_btw_auv_shark_array = []
            # count the number of habitats visited at each step
            num_of_times_visited_habitat_count = 0
            num_unique_habitat_visited_count = 0

            episode_durations.append(max_step)
            traveled_dist_array.append(0.0)

            """dist_btw_auv_shark_array.append(calculate_range(state["auv_pos"], state["shark_pos"]))"""

            eps_reward = 0.0

            for t in range(1, max_step):
                action = self.agent.select_action(state, self.policy_net)

                """dist_btw_auv_shark_array.append(calculate_range(state["auv_pos"], state["shark_pos"]))"""

                reward = self.em.take_action(action)

                traveled_dist_array[eps] += self.em.env.distance_traveled

                eps_reward += reward.item()

                state = self.em.get_state()

                if self.em.done:
                    # because the only way for an episode to terminate is when collision happens,
                    # we can count the number of collisions this way
                    collision_count += 1
                    episode_durations[eps] = t
                    break
               
            print("+++++++++++++++++++++++++++++")
            print("Test Episode # ", eps, "end with reward: ", eps_reward, " used time: ", episode_durations[-1])
            print("+++++++++++++++++++++++++++++")

            total_reward_array.append(eps_reward)

            num_unique_habitat_visited_count = self.em.env.visited_unique_habitat_count

            num_of_times_visited_habitat_count = self.em.env.total_time_in_hab

            # we want to normalize the number of time steps that we have visited an habitat
            #   based on the total distance that the auv has traveled in this episode
            num_of_times_visited_habitat_count_normalized = float(num_of_times_visited_habitat_count)/float(traveled_dist_array[eps])
            num_unique_habitat_visited_count_normalized = float(num_unique_habitat_visited_count)/float(traveled_dist_array[eps])

            time_in_habitat_array.append(num_of_times_visited_habitat_count)
            num_unique_habitat_array.append(num_unique_habitat_visited_count)

            time_in_habitat_array_normalized.append(num_of_times_visited_habitat_count_normalized)
            num_unique_habitat_array_normalized.append(num_unique_habitat_visited_count_normalized)

        self.policy_net.train()

        avg_total_reward = np.mean(total_reward_array)

        collision_rate = float(collision_count) / float(num_episodes) * 100

        # we can normalize it by the average of traveled distance
        collision_rate_normalized = float(collision_rate) / np.mean(traveled_dist_array)

        result = {
            "avg_total_reward": avg_total_reward,
            "avg_dist_btw_auv_shark": [],
            "collision_rate": collision_rate,
            "collision_rate_norm": collision_rate_normalized,
            "avg_timestep_in_hab": np.mean(time_in_habitat_array),
            "avg_timestep_in_hab_norm": np.mean(time_in_habitat_array_normalized),
            "avg_unique_hab_visited": np.mean(num_unique_habitat_array),
            "avg_unique_hab_visited_norm": np.mean(num_unique_habitat_array_normalized),
        }

        return result
    

    def test_q_value_control_auv (self, num_episodes, max_step, live_graph_3D = False, live_graph_2D = False):
        episode_durations = []
        starting_dist_array = []
        traveled_dist_array = []
        final_reward_array = []
        total_reward_array = []

        # if we want to continue training an already trained network
        self.load_trained_network()
        self.policy_net.eval()
        
        for eps in range(num_episodes):
            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly()

            starting_dist = calculate_range(state['auv_pos'], state['shark_pos'])
            starting_dist_array.append(starting_dist)

            episode_durations.append(max_step)
            traveled_dist_array.append(0.0)
            final_reward_array.append(0.0)
            total_reward_array.append(0.0)

            reward = 0

            self.em.env.init_live_graph(live_graph_2D = live_graph_2D)

            for t in range(1, max_step):
                action = self.agent.select_action(state, self.policy_net)
                print("neural network chosen action")
                print(action)
                print("-------")

                print(self.em.possible_actions)
                print("===============")
                v_index = input("linear velocity index: ")
                w_index = input("angular velocity index: ")

                action = torch.tensor([int(v_index), int(w_index)])

                reward = self.em.take_action(action)

                final_reward_array[eps] = reward.item()
                total_reward_array[eps] += reward.item()

                traveled_dist_array[eps] += self.em.env.distance_traveled

                self.em.render(print_state = False, live_graph_3D = live_graph_3D, live_graph_2D = live_graph_2D)

                state = self.em.get_state()

                if self.em.done:
                    episode_durations[eps] = t
                    break
            
            
            print("+++++++++++++++++++++++++++++")
            print("Episode # ", eps, "end with reward: ", reward, " used time: ", episode_durations[-1])
            print("+++++++++++++++++++++++++++++")


        self.em.close()

        print("final sums of time")
        print(episode_durations)
        print("average time")
        print(np.mean(episode_durations))
        print("-----------------")

        text = input("stop")

        print("all the starting distances")
        print(starting_dist_array)
        print("average starting distance")
        print(np.mean(starting_dist_array))
        print("-----------------")

        text = input("stop")

        print("all the traveled distances")
        print(traveled_dist_array)
        print("average traveled dissta")
        print(np.mean(traveled_dist_array))
        print("-----------------")

        text = input("stop")

        print("final reward")
        print(final_reward_array)
        print("-----------------")

        text = input("stop")

        print("total reward")
        print(total_reward_array)
        print("-----------------")
    

    

def main():
    dqn = DQN(N_V, N_W)
    # dqn.train(NUM_OF_EPISODES, MAX_STEP, load_prev_training = True, live_graph_3D = False, live_graph_2D = True, use_HER=False)
    dqn.test(NUM_OF_EPISODES_TEST, MAX_STEP_TEST, live_graph_3D = False, live_graph_2D = True)
    # dqn.test_q_value_control_auv(NUM_OF_EPISODES_TEST, MAX_STEP_TEST, live_graph_3D = False, live_graph_2D = True)


if __name__ == "__main__":
    main()