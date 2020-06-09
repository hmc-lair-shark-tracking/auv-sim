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

from motion_plan_state import Motion_plan_state

# namedtuple allows us to store Experiences as labeled tuples
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

"""
============================================================================

    Parameters

============================================================================
"""

# Define the distance between the auv and the goal
dist = 5.0
MIN_X = dist
MAX_X= dist * 2
MIN_Y = 0.0
MAX_Y = dist * 3

NUM_OF_OBSTACLES = 0

STATE_SIZE = 8 + 4 * NUM_OF_OBSTACLES
ACTION_SIZE = 1

NUM_OF_EPISODES = 250
MAX_STEP = 250

# Most of the hyperparameters come from the DDPG paper
# learning rate
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3

# size of the replay memory
MEMORY_SIZE = 1e6
BATCH_SIZE = 64

# use GPU if available, else use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# number of additional goals to be added to the replay memory
NUM_GOALS_SAMPLED_HER = 4

# discount factor for calculating target q values
GAMMA = 0.99
TAU = 0.001

SAVE_EVERY = 10

DEBUG = True

"""
============================================================================

    Helper Functions

============================================================================
"""

def process_state_for_nn(state):
    """
    Convert the state (observation in the environment) to a tensor so it can be passed into the neural network

    Parameter:
        state - a tuple of two np arrays
            Each array is this form [x, y, z, theta]
    """
    auv_tensor = torch.from_numpy(state[0])
    shark_tensor = torch.from_numpy(state[1])
    obstacle_tensor = torch.from_numpy(state[2])
    obstacle_tensor = torch.flatten(obstacle_tensor)
    
    # join 2 tensor together
    return torch.cat((auv_tensor, shark_tensor, obstacle_tensor)).float()
    


def init_weight_by_fanin(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)



def extract_tensors(experiences):
    """
    Convert batches of experiences sampled from the replay memeory to tuples of tensors
    """
    batch = Experience(*zip(*experiences))
   
    t1 = torch.stack(batch.state)
    t2 = torch.stack(batch.action)
    t3 = torch.stack(batch.reward)
    t4 = torch.stack(batch.next_state)
    t5 = torch.stack(batch.done)

    return (t1, t2, t3, t4, t5)



def save_model(actor, actor_target, critic, critic_target):
    print("Model Save...")
    torch.save(actor.state_dict(), 'checkpoint_actor.pth')
    torch.save(actor_target.state_dict(), 'checkpoint_actor_target.pth')
    torch.save(critic.state_dict(), 'checkpoint_critic.pth')
    torch.save(critic_target.state_dict(), 'checkpoint_critic_target.pth')



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



def generate_rand_obstacles(auv_init_pos, shark_init_pos, num_of_obstacles):
    """
    """
    obstacle_array = []
    for i in range(num_of_obstacles):
        obs_x = np.random.uniform(MIN_X, MAX_X)
        obs_y = np.random.uniform(MIN_Y, MAX_Y)
        obs_size = np.random.randint(1,5)
        while validate_new_obstacle([obs_x, obs_y], obs_size, auv_init_pos, shark_init_pos, obstacle_array):
            obs_x = np.random.uniform(MIN_X, MAX_X)
            obs_y = np.random.uniform(MIN_Y, MAX_Y)
        obstacle_array.append(Motion_plan_state(x = obs_x, y = obs_y, z=-5, size = obs_size))

    return obstacle_array



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

    Classes

============================================================================
"""


"""
Class for building policy and target neural network
"""
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden1 = 400, hidden2 = 300, init_w = 3e-3):
        """
        Initialize the Q neural network with input

        Parameter:
            
        """
        super().__init__()
    
        # input layer
        self.fc1 = nn.Linear(in_features = state_size, out_features = hidden1)
        self.bn1 = nn.LayerNorm(hidden1)

        # branch for selecting v
        # TODO: for now, maybe let's not try branching?
        # myabe the action_size means how many differnt action we have to take
        # in DDPG paper, it mentions to include action at the 2nd hidden layer of Q
        self.fc2 = nn.Linear(in_features = hidden1, out_features = hidden2) 
        self.bn2 = nn.LayerNorm(hidden2)  
        self.out = nn.Linear(in_features = hidden2, out_features = action_size)

        self.init_weight(init_w)

        # branch for selecting w
        # self.fc2_w = nn.Linear(in_features = hidden1, out_features = hidden2)
        # self.bn2_w = nn.LayerNorm(hidden2)     
        # self.out_w = nn.Linear(in_features = , out_features=action_size)
    

    def init_weight(self, init_w):
        # initialize the rest of the layers with a unifrom distribution of "1/sqrt(f)"
        #   where f is the fan-in of the layer
        self.fc1.weight.data = init_weight_by_fanin(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight_by_fanin(self.fc2.weight.data.size())
        # initialize the final layer with a uniform distribution of "init_w"
        self.out.weight.data.uniform_(-init_w, init_w)


    def forward(self, state):
        """
        Define the forward pass through the neural network

        Parameters:
            
        """
        # pass through the layers then have relu applied to it
        # relu is the activation function that will turn any negative value to 0,
        #   and keep any positive value
        
        action = self.fc1(state)

        # print("first layer")
        # print(action)
        # text = input("stop")

        action = F.relu(action)

        # print("first layer relu")
        # print(action)
        # text = input("stop")

        action = self.bn1(action)      # batch normalization
        
        # print("first layer batch norm")      
        # print(action)
        # text = input("stop")
        
        action = self.fc2(action)

        # print("second layer")
        # print(action)
        # text = input("stop")

        action = F.relu(action)

        # print("second layer relu")
        # print(action)
        # text = input("stop")

        action = self.bn2(action)       # batch normalization

        # print("second layer batch norm")
        # print(action)
        # text = input("stop")
        
        # TODO: debating whether to clip the output here
        # I worry that by clipping the output, it will affect back propagation
        action = self.out(action)
        print("---------------------")
        print("before being tanh")
        print(action)
        print("---------------------")

        action = torch.tanh(action)

        print("output layer")
        print(action)
        text = input("stop")

        return action



"""
Neural Net to map state-action pairs to Q-values
"""
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden1 = 400, hidden2 = 300, init_w = 3e-3):
        """
        Initialize the Q neural network with input

        Parameter:
            
        """
        super().__init__()
    
        # input layer
        self.fc1 = nn.Linear(in_features = state_size, out_features = hidden1)
        self.bn1 = nn.LayerNorm(hidden1)

        # branch for selecting v
        # TODO: for now, maybe let's not try branching?
        # myabe the action_size means how many differnt action we have to take
        # in DDPG paper, it mentions to include action at the 2nd hidden layer of Q
        self.fc2 = nn.Linear(in_features = hidden1 + action_size, out_features = hidden2) 
        self.bn2 = nn.LayerNorm(hidden2)  
        self.out = nn.Linear(in_features = hidden2, out_features = 1)

        self.init_weight(init_w)

        # branch for selecting w
        # self.fc2_w = nn.Linear(in_features = hidden1, out_features = hidden2)
        # self.bn2_w = nn.LayerNorm(hidden2)     
        # self.out_w = nn.Linear(in_features = , out_features=action_size)
    

    def init_weight(self, init_w):
        # initialize the rest of the layers with a unifrom distribution of "1/sqrt(f)"
        #   where f is the fan-in of the layer
        self.fc1.weight.data = init_weight_by_fanin(self.fc1.weight.data.size())
        self.fc2.weight.data = init_weight_by_fanin(self.fc2.weight.data.size())
        # initialize the final layer with a uniform distribution of "init_w"
        self.out.weight.data.uniform_(-init_w, init_w)


    def forward(self, state, action):
        """
        Define the forward pass through the neural network

        Parameters:
            t - the state as a tensor
        """
        # pass through the layers then have relu applied to it
        # relu is the activation function that will turn any negative value to 0,
        #   and keep any positive value

        # print("input state")
        # print(state)
        
        q_val = self.fc1(state)

        # print("first layer")
        # print(q_val)
        # text = input("stop")

        q_val = F.relu(q_val)

        # print("first layer relu")
        # print(q_val)
        # text = input("stop")

        q_val = self.bn1(q_val)             # batch normalization

        # print("first layer bn")
        # print(q_val)
        # print(q_val.size())
        # print(q_val.type())
        # print("input action")
        # print(action)
        # print(action.size())
        # print(action.type())
        # text = input("stop")

        # introduce action into the hidden layer
        q_val = torch.cat((q_val, action), dim=1)
        # print("dimension")
        # print(q_val.size())
        # text = input("stop")

        # print("combine action and q_val")
        # print(q_val)
        # text = input("stop")

        q_val = self.fc2(q_val)

        # print("second layer")
        # print(q_val)
        # text = input("stop")

        q_val = F.relu(q_val)

        # print("second layer relu")
        # print(q_val)
        # text = input("stop")

        q_val = self.bn2(q_val)       # batch normalization

        # print("second layer bn")
        # print(q_val)
        # text = input("stop")

        q_val = self.out(q_val)

        # print("output layer")
        # print(q_val)
        # text = input("stop")

        return q_val



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
Based on the Ornstein-Uhlenbeck process
Provide a noise process to our actor policy (solve the problem of exploration vs exploitation)

from https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ounoise.py
"""
class OU_Noise():
    def __init__(self, action_dimension, scale = 0.1, mu = 0.0, theta = 0.15, sigma = 0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale



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
        self.env = gym.make('gym_auv:auv-v0').unwrapped

        self.current_state = None
        self.done = False

        # self.min_v, self.min_w = self.env.action_space.low
        # self.max_v, self.max_w = self.env.action_space.high
        self.min_theta = self.env.action_space.low
        self.max_theta = self.env.action_space.high


    def init_env_randomly(self):
        auv_init_pos = Motion_plan_state(x = np.random.uniform(MIN_X, MAX_X), y = np.random.uniform(MIN_X, MAX_X), z = -5.0, theta = 0)
        shark_init_pos = Motion_plan_state(x = np.random.uniform(MIN_Y, MAX_Y), y = np.random.uniform(MIN_Y, MAX_Y), z = -5.0, theta = 0) 
        obstacle_array = generate_rand_obstacles(auv_init_pos, shark_init_pos, NUM_OF_OBSTACLES)

        if DEBUG:
            print("===============================")
            print("Starting Positions")
            print(auv_init_pos)
            print(shark_init_pos)
            print(obstacle_array)
            print("===============================")
            text = input("stop")

        return self.env.init_env(auv_init_pos, shark_init_pos, obstacle_array)


    def reset(self):
        """
        Reset the environment and return the initial state
        """
        return self.env.reset()


    def close(self):
        self.env.close()


    def render(self, mode='human', print_state = True, live_graph = False):
        """
        Render the environment both as text in terminal and as a 3D graph if necessary

        Parameter:
            mode - string, modes for rendering, currently only supporting "human"
            live_graph - boolean, will display the 3D live_graph if True
        """
        state = self.env.render(mode, print_state)
        if live_graph: 
            self.env.render_3D_plot(state[0], state[1])
        return state


    def num_actions_available(self):
        """
        Return the number of options (values) to choose for v and w
        """
        return len(self.possible_actions[0])

    
    def clip_value_to_range(self, value, target_min, target_max):
        """
        From: https://stackoverflow.com/questions/49911206/how-to-restrict-output-of-a-neural-net-to-a-specific-range
        """
        # tanh gives you range between -1 and 1, so this gives you range between 0, 2
        # new_value = value + 1
        new_value = np.tanh(value) + 1 
        scale = (target_max - target_min) / 2.0
        return new_value * scale + target_min


    def take_action(self, action):
        """
        Parameter: 
            action - tensor of the format: tensor([v_index, w_index])
                use the index from the action and take a step in environment
                based on the chosen values for v and w
        """
        # v_action_raw = action[0].item()
        # w_action_raw = action[1].item()
        theta_action = action.item()

        # clip the action so that they are within the range
        # TODO: using clip might have potential problem
        #   (when it starts out, it might stay at the min for a long time even though the neural net output is changing slightly)
        # v_action = np.clip(v_action_raw, self.min_v, self.max_v)
        # w_action = np.clip(w_action_raw, self.min_w, self.max_w)

        # v_action = self.clip_value_to_range(v_action_raw, self.min_v, self.max_v)
        # w_action = self.clip_value_to_range(w_action_raw, self.min_w, self.max_w)

        # theta_action = angle_wrap(theta_action_raw)
       
        # we only care about the reward and whether or not the episode has ended
        # action is a tensor, so item() returns the value of a tensor (which is just a number)
        self.current_state, reward, self.done, _ = self.env.step(theta_action)

        if DEBUG:
            print("=========================")
            # print("action v: ", v_action_raw, " | ", v_action)  
            # print("action w: ", w_action_raw, " | ", w_action)  
            print("action theta: ", theta_action)
            print("new state: ")
            print(self.current_state)
            print("reward: ")
            print(reward)
            print("=========================")
            # text = input("stop")

        # wrap reward into a tensor, so we have input and output to both be tensor
        return torch.tensor([reward], device=self.device).float()


    def get_state(self):
        """
        state will be represented as the difference bewteen 2 screens
            so we can calculate the velocity
        """
        return self.env.state


    def get_binary_reward(self, auv_pos, goal_pos):
        """
        Wrapper to convert the binary reward (-1 or 1) to a tensor

        Parameters:
            auv_pos - an array of the form [x, y, z, theta]
            goal_pos - an array of the same form, represent the target position that the auv is currently trying to reach
        """
        reward = self.env.get_binary_reward(auv_pos, goal_pos)

        return torch.tensor([reward], device=self.device).float()



class DDPG():
    def __init__(self, state_size, action_size):
        # initialize the actor and critic network
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        # initialize the optimizer of the actor and critic network with the corresponding learning rate
        self.actor_optimizer = optim.Adam(params = self.actor.parameters(), lr = LR_ACTOR)
        self.critic_optimizer = optim.Adam(params = self.critic.parameters(), lr = LR_CRITIC)

        # intialize the target networks for the actor-critic network
        self.actor_target = Actor(state_size, action_size)
        self.critic_target= Critic(state_size, action_size)
        # the target networks will start out with the same weights as the actor-critic network
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.memory = ReplayMemory(MEMORY_SIZE)

        # set up the environment
        self.em = AuvEnvManager(DEVICE)

        # set up the noise process
        self.noise = OU_Noise(action_dimension = action_size)
    

    def hard_update(self, target, source):
        """
        Make sure that the target have the same parameters as the source
            Used to initialize the target networks for the actor and critic
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    
    def soft_update(self, local_model, target_model, tau):
        """
        From https://github.com/tobiassteidle/Reinforcement-Learning/blob/master/OpenAI/MountainCarContinuous-v0/Agent.py
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def load_trained_network(self):
        """
        Load already trained neural network
        """
        self.actor.load_state_dict(torch.load('checkpoint_policy.pth'))
        self.critic.load_state_dict(torch.load('checkpoint_target.pth'))


    def select_action(self, state, add_noise = True):
        # TODO: verify that this helper function still works in DDGP
        # convert the state to a tensor so it can get passed into the neural net
        state = process_state_for_nn(state)

        # set the actor nn to evaluation mode
        self.actor.eval()

        with torch.no_grad():
            action = self.actor(state).to(DEVICE)

        # set the actor nn back to train mode
        self.actor.train()

        if DEBUG:
            print("action without noise: ")
            print(action)

        if add_noise:
            action = action + self.noise.noise()
        
        # apparently, this is giving as double while everything else is as float
        # cast the action to float to make pytorch happy
        return action.float()

    
    def possible_extra_goals(self, time_step, next_state_array):
        # currently, we use the "future" strategy mentioned in the HER paper
        #   replay with k random states which come from the same episode as the transition being replayed and were observed after it
        her_strategy = "final"
       
        additional_goals = []

        if her_strategy == "future":
            possible_goals_to_sample = next_state_array[time_step+1: ]

            # only sample additional goals if there are enough to sample
            # TODO: slightly modified from our previous implementation of HER, maybe this is better?
            if len(possible_goals_to_sample) >= NUM_GOALS_SAMPLED_HER:
                additional_goals = random.sample(possible_goals_to_sample, k = NUM_GOALS_SAMPLED_HER)
        elif her_strategy == "final":
            additional_goals.append(next_state_array[-1])
        
        return additional_goals


    def store_extra_goals_HER(self, action, state, next_state, additional_goals):
        """print("------------------------")
        print("additional experiences HER")"""
        for goal in additional_goals:
            # build new current state and new next state based on the new goal
            new_curr_state = (state[0], goal[0], state[2])

            new_next_state = (next_state[0], goal[0], next_state[2])

            reward = self.em.get_binary_reward(new_next_state[0], new_next_state[1])

            done = torch.tensor([False], device=DEVICE)
            if reward.item() == 1:
                done = torch.tensor([True], device=DEVICE)

            self.memory.push(Experience(process_state_for_nn(new_curr_state), action, process_state_for_nn(new_next_state), reward, done))
            """print(Experience(process_state_for_nn(new_curr_state), action, process_state_for_nn(new_next_state), reward, done))"""


    def update_neural_nets(self):
        if self.memory.can_provide_sample(BATCH_SIZE):
            # sample a random minibatch of "BATCH_SIZE" experiences
            experiences_batch = self.memory.sample(BATCH_SIZE)
            
            """
            print("--------------------")
            print("sampled experiences")
            print(experiences_batch)
            text = input("stop")
            """

            # extract states, actions, rewards, next_states into their own individual tensors from experiences batch
            states_batch, actions_batch, rewards_batch, next_states_batch, done_batch = extract_tensors(experiences_batch)
            
            """
            print("----")
            print("states")
            print(states_batch)
            print("----")
            print("actions")
            print(actions_batch)
            print("----")
            print("rewards")
            print(rewards_batch)
            print("----")
            print("next states")
            print(next_states_batch)
            print("----")
            print("done")
            print(done_batch)

            text = input("stop")
            """
            
            # --------------------- Update the Critic Network ---------------------

            # get the predicted actions based on next states
            next_actions_batch = self.actor_target(next_states_batch)

            """
            print("****************************")
            print("next action batch")
            print(next_actions_batch)
            text = input("stop")
            """

            # get the predicted Q-values based on the next states and actions pairs
            next_target_q_val_batch = self.critic_target(next_states_batch, next_actions_batch)

            """
            print("****************************")
            print("next target q val batch based on next state-action pair")
            print(next_target_q_val_batch)
            text = input("stop")
            """

            # compute the current Q values using the Bellman equation
            target_q_val_batch = (rewards_batch + GAMMA * next_target_q_val_batch)

            """
            print("****************************")
            print("current q val batch")
            print(target_q_val_batch)
            text = input("stop")
            """
            
            # compute the expected q value based on the critic neural net
            expected_q_val_batch = self.critic(states_batch, actions_batch)

            """
            print("****************************")
            print("expected q value based on the critic")
            print(expected_q_val_batch)
            text = input("stop")
            """
            
            # calculate the loss for the critic neural net by using mean square error
            critic_loss = F.mse_loss(expected_q_val_batch, target_q_val_batch)

            """
            print("****************************")
            print("critic loss")
            print(critic_loss)
            text = input("stop")
            """

            # update the critic neural net based on the loss
            
            # set the gradients of all the weights and bias in the critic neural net to zero
            #   need to zero out the gradients, so we don't accumulate gradients across all back propogation
            self.critic_optimizer.zero_grad()   
            # computes the gradient of critic loss with respect to all the weights and biases in critic neural net
            critic_loss.backward()
            # updates the weights and biases in the cirtic neural net with the gradients that we just calculated
            self.critic_optimizer.step()

            # --------------------- Update the Actor Neural Network ---------------------
            """print("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")
            print("=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")"""

            predicted_actions_batch = self.actor(states_batch)

            """print("****************************")
            print("predicted action based on actor")
            print(predicted_actions_batch)
            text = input("stop")"""

            actor_loss = -self.critic(states_batch, predicted_actions_batch).mean()

            """print("****************************")
            print("actor loss")
            print(actor_loss)
            text = input("stop")"""

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # --------------------- Update the Target Networks ---------------------
            self.soft_update(self.critic, self.critic_target, TAU)
            self.soft_update(self.actor, self.actor_target, TAU)

            
            self.actor_loss_in_ep.append(actor_loss.item())
            self.critic_loss_in_ep.append(critic_loss.item())


    def train(self, num_episodes, max_step, load_prev_training=False):
        # keep track of how many steps the auv takes in each episode
        self.episode_durations = []
        self.actor_loss_in_training = []
        self.critic_loss_in_training = []

        if load_prev_training:
            # if we want to continue training an already trained network
            self.load_trained_network()
        
        for eps in range(num_episodes):
            # initialize a random noise process N for action exploration 
            self.noise.reset()

            # initialize the starting point of the shark and the auv randomly
            # receive initial observation state s1 
            state = self.em.init_env_randomly()

            score = 0

            action_array = []
            next_state_array = []

            self.actor_loss_in_ep = []
            self.critic_loss_in_ep = []

            # determine how many steps we should run HER
            # by default, it will be "max_step" - 1 because in the first loop, we start at t=1
            iteration = max_step - 1

            for t in range(1, max_step):
                # Select action according to the current policy and exploration noise
                action = self.select_action(state)

                # store the action for HER algorithm
                action_array.append(action)

                # Execute action and observe reward + new state
                self.em.take_action(action)
                
                next_state = self.em.get_state()
                next_state_array.append(next_state)

                self.em.render(print_state = False, live_graph=True)

                state = next_state

                if self.em.done:
                    # if the auv has reached the goal
                    # modify how many steps we should run HER
                    iteration = t
                    break

            # restart the starting state to prepare for HER algorithm
            state = self.em.reset()

            self.episode_durations.append(iteration)

            if DEBUG:
                step = input("stop")
            
            for t in range(iteration):
                action = action_array[t]
                next_state = next_state_array[t]

                # next_state[0] - the auv position after it has taken an action
                # next_state[1] - the actual goal
                reward = self.em.get_binary_reward(next_state[0], next_state[1])

                done = torch.tensor([False], device=DEVICE)
                if reward.item() == 1:
                    done = torch.tensor([True], device=DEVICE)

                # store the actual experience in the memory
                self.memory.push(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward, done))

                """
                print("----------------------------")
                print("timestep: ", t)
                print("actual experience stored")
                print(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward, done))
                print("----------------------------")
                text = input("stop")
                """

                additional_goals = self.possible_extra_goals(t, next_state_array)
                self.store_extra_goals_HER(action, state, next_state, additional_goals)

                state = next_state

                self.update_neural_nets()    

            if eps % SAVE_EVERY == 0 or self.em.done:
                save_model(self.actor, self.actor_target, self.critic, self.critic_target)

            print("+++++++++++++++++++++++++++++++++++++++++")
            print("Episode # ", eps, " used time: ", self.episode_durations[-1])

            if self.actor_loss_in_ep != []:
                avg_actor_loss = np.mean(self.actor_loss_in_ep)
                self.actor_loss_in_training.append(avg_actor_loss)
                print("average actor loss: ", avg_actor_loss)
            else:
                self.actor_loss_in_training.append(1000)
            
            if self.critic_loss_in_ep != []:
                avg_critic_loss =  np.mean(self.critic_loss_in_ep)
                self.critic_loss_in_training.append(avg_critic_loss)
                print("average critic loss: ", avg_critic_loss)
            else:
                self.critic_loss_in_training.append(1000)

            print("+++++++++++++++++++++++++++++++++++++++++")

            if DEBUG:
                text = input("stop")

        print("steps in each episode")
        print(self.episode_durations)
        print("actor loss")
        print(self.actor_loss_in_training)
        print("critic loss")
        print(self.critic_loss_in_training)
        


def train():
    ddpg = DDPG(STATE_SIZE, ACTION_SIZE)
    ddpg.train(NUM_OF_EPISODES, MAX_STEP)


    
def main():
    train()
    # test_trained_model()

if __name__ == "__main__":
    main()