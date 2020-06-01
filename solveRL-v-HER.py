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

MIN_X = 0.0
MAX_X= 100.0
MIN_Y = 0.0
MAX_Y = 100.0

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
    

"""
Class for building policy and target neural network
"""
class DQN(nn.Module):
    def __init__(self, input_size, output_size_v, output_size_w):
        """
        Initialize the Q neural network with input

        Parameter:
            input_size - int, the size of observation space
            output_size_v - int, the number of possible options for v
            output_size_y - int, the number of possible options for w
        """
        super().__init__()

        # 2 fully connected hidden layers
        # first layer will have "input_size" inputs
        #   Cureently, auv 3D position and theta + shark 3D postion and theta
        self.fc1 = nn.Linear(in_features=input_size, out_features=24)  
        # branch for selecting v
        self.fc2_v = nn.Linear(in_features=24, out_features=32)      
        self.out_v = nn.Linear(in_features=32, out_features=output_size_v)

        # branch for selecting w
        self.fc2_w = nn.Linear(in_features=24, out_features=32)
        self.out_w = nn.Linear(in_features=32, out_features=output_size_w)


    def forward(self, t):
        """
        Define the forward pass through the neural network

        Parameters:
            t - the state as a tensor
        """

        # pass through the layers then have relu applied to it
        # relu is the activation function that will turn any negative value to 0,
        #   and keep any positive value
        t = self.fc1(t)
        t = F.relu(t)

        # the neural network is separated into 2 separate branch
        t_v = self.fc2_v(t)
        t_v = F.relu(t_v)

        t_w = self.fc2_v(t)
        t_w = F.relu(t_w)

        # pass through the last layer, the output layer
        # output is a tensor of Q-Values for all the optinons for v/w
        t_v = self.out_v(t_v)
        t_w = self.out_w(t_w)

        return torch.stack((t_v, t_w))


# namedtuple allows us to store Experiences as labeled tuples
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


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
    def __init__(self, strategy, actions_range_v, actions_range_w, device):
        # the agent's current step in the environment
        self.current_step = 0
        # in our case, the strategy will be the epsilon greedy strategy
        self.strategy = strategy
        # how many possible actions can the agent take at a given state
        # for our case, it only has 2 actions (left or right)
        self.actions_range_v = actions_range_v
        self.actions_range_w = actions_range_w
        # what we want to PyTorch to use for tensor calculation
        self.device = device

        self.ra = None
        self.ea = None

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            # print("-----")
            # print("randomly picking")
            v_action_index = random.choice(range(self.actions_range_v))
            w_action_index = random.choice(range(self.actions_range_w))

            return torch.tensor([v_action_index, w_action_index]).to(self.device) # explore  
        else:
            # turn off gradient tracking bc we are using the model for inference instead of training
            # we don't need to keep track the gradient because we are not doing backpropagation to figure out the weight 
            # of each node yet
            with torch.no_grad():
                # for the given "state"，the output will be the action 
                #   with the highest Q-Value output from the policy net
                # print("-----")
                # print("exploiting")
                state = process_state_for_nn(state)

                output_weight = policy_net(state).to(self.device)

                v_action_index = torch.argmax(output_weight[0]).item()
                w_action_index = torch.argmax(output_weight[1]).item()

                return torch.tensor([v_action_index, w_action_index]).to(self.device) # explore  


class AuvEnvManager():
    def __init__(self, device, N, auv_init_pos, shark_init_pos, obstacle_array = []):
        self.device = device
        # have access to behind-the-scenes dynamics of the environment 
        self.env = gym.make('gym_auv:auv-v0').unwrapped
        self.env.init_env(auv_init_pos, shark_init_pos, obstacle_array)

        self.current_state = None
        self.done = False

        self.possible_actions = self.env.actions_range(N)

    def reset(self):
        self.env.reset()
        
    def close(self):
        self.env.close()
    
    def render(self, mode='human', live_graph = False):
        state = self.env.render(mode)
        if live_graph: 
            self.env.render_3D_plot(state[0], state[1])
        return state

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        v_action_index = action[0].item()
        w_action_index = action[1].item()
        v_action = self.possible_actions[0][v_action_index]
        w_action = self.possible_actions[1][w_action_index]
        # print("=========================")
        # print("action v: ", v_action_index, " | ", v_action)  
        # print("action w: ", w_action_index, " | ", w_action)  
        
        # we only care about the reward and whether or not the episode has ended
        # action is a tensor, so item() returns the value of a tensor (which is just a number)
        self.current_state, reward, self.done, _ = self.env.step((v_action, w_action))
        # print("new state: ")
        # print(self.current_state)
        # print("reward: ")
        # print(reward)
        # print("=========================")

        # wrap reward into a tensor, so we have input and output to both be tensor
        return torch.tensor([reward], device=self.device).float()

    def get_state(self):
        """
        state will be represented as the difference bewteen 2 screens
            so we can calculate the velocity
        """
        return self.env.state


# Ultility functions
def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=0).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()        
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)    
    plt.pause(0.001)
    print("Episode", len(values), "\n", \
        moving_avg_period, "episode moving avg:", moving_avg[-1])


def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))


    t1 = torch.stack(batch.state)
    t2 = torch.stack(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.stack(batch.next_state)

    return (t1,t2,t3,t4)


class QValues():
    """
    mainly use its static method
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # actions is a tensor with this format: [[v_action_index1, w_action_index1], [v_action_index2, w_action_index2] ]
        # actions[:,:1] gets only the first element in the [v_action_index, w_action_index], 
        #   so we get all the v_action_index as a tensor
        # policy_net(states) gives all the predicted q-values for all the action outcome for a given state
        # policy_net(states).gather(dim=1, index=actions[:,:1]) gives us
        #   a tensor of the q-value corresponds to the state and action(specified by index=actions[:,:1]) pair 
        # print(policy_net(states))
        
        q_values_for_v = policy_net(states)[0].gather(dim=1, index=actions[:,:1])
        q_values_for_w = policy_net(states)[1].gather(dim=1, index=actions[:,1:2])
       
        return torch.stack((q_values_for_v, q_values_for_w), dim = 0)

    
    @staticmethod        
    def get_next(target_net, next_states):  
        # for each next state, we want to obtain the max q-value predicted by the target_net among all the possible next actions              
        # we want to know where the final states are bc we shouldn't pass them into the target net

        # check individual next state's max value
        # if max value is 0 (.eq(0)), set to be true
        # final_state_locations = next_states.flatten(start_dim=1) \
        #     .max(dim=1)[0].eq(0).type(torch.bool)
        # flip the non final_state 
        # non_final_state_locations = (final_state_locations == False)

        # non_final_states = next_states[non_final_state_locations]

        # batch_size = next_states.shape[0]
        # # create a tensor of zero
        # values = torch.zeros(batch_size).to(QValues.device)

        # # a tensor of 
        # #   zero - if it's a final state
        # #   target_net's maximum predicted q-value - if it's a non-final state.
        # values[non_final_state_locations] = target_net(non_final_states).max(dim=0)[0].detach()
        # return values
       
        v_max_q_values = target_net(next_states)[0].max(dim=1)[0].detach()
        w_max_q_values = target_net(next_states)[1].max(dim=1)[0].detach()

        # print(v_max_q_values)
        # print(w_max_q_values)
        # this will allow you to pair element by element!
        # print(torch.stack((v_max_q_values, w_max_q_values), dim = 0))
       
        return torch.stack((v_max_q_values, w_max_q_values), dim = 0)


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
    auv_overlaps = calculate_range([auv_init_pos.x, auv_init_pos.y], new_obstacle) <= new_obs_size
    shark_overlaps = calculate_range([shark_init_pos.x, shark_init_pos.y], new_obstacle) <= new_obs_size
    obs_overlaps = False
    for obs in obstacle_array:
        if calculate_range([obs.x, obs.y], new_obstacle) <= (new_obs_size + obs.size):
            obs_overlaps = True
            break
    return auv_overlaps or shark_overlaps or obs_overlaps

def generate_rand_obstacles(auv_init_pos, shark_init_pos, num_of_obstacles):
    obstacle_array = []
    for i in range(num_of_obstacles):
        obs_x = np.random.uniform(MIN_X, MAX_X)
        obs_y = np.random.uniform(MIN_Y, MAX_Y)
        obs_size = np.random.randint(1,11)
        while validate_new_obstacle([obs_x, obs_y], obs_size, auv_init_pos, shark_init_pos, obstacle_array):
            obs_x = np.random.uniform(MIN_X, MAX_X)
            obs_y = np.random.uniform(MIN_Y, MAX_Y)
        obstacle_array.append(Motion_plan_state(x = obs_x, y = obs_y, z=-5, size = obs_size))

    return obstacle_array


def train():
    batch_size = 256
    # discount factor for exploration rate decay
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001

    # how frequently (in terms of episode) we will update the target policy network with 
    #   weights from the policy network
    target_update = 10

    # capacity of replay memory
    memory_size = 100000

    # learning rate
    lr = 0.001

    num_episodes = 300

    # use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameter to discretize the action v and w
    # N specify the number of options that we get to have for v and w
    N = 8

    num_of_obstacles = 2

    auv_init_pos = Motion_plan_state(x = np.random.uniform(0.0, 300.0), y = np.random.uniform(0.0, 500.0), z = -5.0, theta = 0)
    shark_init_pos = Motion_plan_state(x = np.random.uniform(0.0, 300.0), y = np.random.uniform(0.0, 500.0), z = -5.0, theta = 0) 
    obstacle_array = generate_rand_obstacles(auv_init_pos, shark_init_pos, num_of_obstacles)
    
    em = AuvEnvManager(device, N, auv_init_pos, shark_init_pos, obstacle_array)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy, N, N, device)
    memory = ReplayMemory(memory_size)

    input_size = 8 + len(obstacle_array) * 4

    # to(device) puts the network on our defined device
    policy_net_v = DQN(input_size, N, N).to(device)
    target_net_v = DQN(input_size, N, N).to(device)

    # set the weight and bias in the target_net to be the same as the policy_net
    target_net_v.load_state_dict(policy_net_v.state_dict())
    # set the target_net in evaluation mode instead of training mode (bc we are only using it to 
    # estimate the next max Q value)
    target_net_v.eval()

    optimizer = optim.Adam(params=policy_net_v.parameters(), lr=lr)

    episode_durations = []

    save_every = 20

    max_step = 3000

    def save_model():
        print("Model Save...")
        torch.save(policy_net_v.state_dict(), 'checkpoint_policy.pth')
        torch.save(target_net_v.state_dict(), 'checkpoint_target.pth')

    # For each episode:
    for episode in range(num_episodes):
        # randomize the auv and shark position
        auv_init_pos = Motion_plan_state(x = np.random.uniform(MIN_X, MAX_X), y = np.random.uniform(MIN_Y, MAX_Y), z = -5.0, theta = 0)
        shark_init_pos = Motion_plan_state(x = np.random.uniform(MIN_X, MAX_X), y = np.random.uniform(MIN_Y, MAX_Y), z = -5.0, theta = 0) 
        obstacle_array = generate_rand_obstacles(auv_init_pos, shark_init_pos, num_of_obstacles)

        em.env.init_env(auv_init_pos, shark_init_pos, obstacle_array)
        print("===============================")
        print("Inital State")
        print(auv_init_pos)
        print(shark_init_pos)
        print(obstacle_array)
        print("===============================")
        
        # Initialize the starting state.
        em.reset()
        
        state = em.get_state()
        score = 0
        timestep = time.time()

        for timestep in range(max_step): 
            # For each time step:
            # Select an action (Via exploration or exploitation)
            action = agent.select_action(state, policy_net_v)
            
            # Execute selected action in an emulator.
            # Observe reward and next state.
            reward = em.take_action(action)

            em.render(live_graph=True)

            score += reward.item()

            next_state = em.get_state()
            
            # Store experience in replay memory.
            memory.push(Experience(process_state_for_nn(state), action, process_state_for_nn(next_state), reward))

            state = next_state

            if memory.can_provide_sample(batch_size):
                # Sample random batch from replay memory.
                experiences = memory.sample(batch_size)

                # extract states, actions, rewards, next_states into their own individual tensors from experiences batch
                states, actions, rewards, next_states = extract_tensors(experiences)

 
                # Pass batch of preprocessed states to policy network.
                # return the q value for the given state-action pair by passing throught the policy net
                current_q_values = QValues.get_current(policy_net_v, states, actions)

                next_q_values = QValues.get_next(target_net_v, next_states)
                
                target_q_values_v = (next_q_values[0] * gamma) + rewards
                target_q_values_w = (next_q_values[1] * gamma) + rewards
                # print(next_q_values[0])
                # print(target_q_values_v)
                # print("==================")   
                # print(next_q_values[1])
                # print(target_q_values_w)
                # print(rewards))
                # print(rewards.size())
                # print("==================")   
                # print(rewards.unsqueeze(0))
                # print(rewards.unsqueeze(0).size())
                # print(current_q_values[0].size())
                # print(target_q_values_v.unsqueeze(1).size())

                # Calculate loss between output Q-values and target Q-values.
                # mse_loss calculate the mean square error
                loss_v = F.mse_loss(current_q_values[0], target_q_values_v.unsqueeze(1))
                loss_w = F.mse_loss(current_q_values[1], target_q_values_w.unsqueeze(1))

                loss_total = loss_v + loss_w
                
                # Gradient descent updates weights in the policy network to minimize loss.
                # sets the gradients of all the weights and biases in the policy network to zero
                # so that we can do back propagation 
                optimizer.zero_grad()

                # use backward propagation to calculate the gradient of loss with respect to all the weights and biases in the policy net
                loss_total.backward()

                # updates the weights and biases of all the nodes based on the gradient
                optimizer.step()
            
            if em.done: 
                episode_durations.append(timestep)
                # plot(episode_durations, 100)
                break

            

        #  After x time steps, weights in the target network are updated to the weights in the policy network.
        # in our case, it will be 10 episodes
        if episode % target_update == 0:
            target_net_v.load_state_dict(policy_net_v.state_dict())

        print("+++++++++++++++++++++++++++++")
        print(em.get_state())
        print("Episode # ", episode, "end with reward: ", score, " used time: ", timestep)
        print("+++++++++++++++++++++++++++++")

        if episode % save_every == 0:
            save_model()

        if score >= 13.5:
            save_model()

        time.sleep(1)

    em.close()
    print(episode_durations)


def test_trained_model():
    batch_size = 256
    # discount factor for exploration rate decay
    gamma = 0.999
    eps_start = 0.05
    eps_end = 0.01
    eps_decay = 0.001

    # learning rate
    lr = 0.001

    # use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameter to discretize the action v and w
    # N specify the number of options that we get to have for v and w
    N = 4

    auv_init_pos = Motion_plan_state(x = 740.0, y = 280.0, z = -5.0, theta = 0)
    shark_init_pos = Motion_plan_state(x = 750.0, y = 280.0, z = -5.0, theta = 0) 
    obstacle_array = []
    
    em = AuvEnvManager(device, N, auv_init_pos, shark_init_pos, obstacle_array)

    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy, N, N, device)

    # to(device) puts the network on our defined device
    policy_net_v = DQN(8, N, N).to(device)
    target_net_v = DQN(8, N, N).to(device)

    # if we want to load the already trained network
    policy_net_v.load_state_dict(torch.load('checkpoint_policy.pth'))
    target_net_v.load_state_dict(torch.load('checkpoint_target.pth'))

    # policy_net_v.eval()
    # target_net_v.eval()

    time_list = []
    for i in range(3):
        print("start testing the network")
        em.reset()
        state = em.get_state()
        em.env.init_data_for_3D_plot(auv_init_pos, shark_init_pos)
        time_list.append(0)
        for t in range(1500):
            action = agent.select_action(state, policy_net_v)
            em.render(live_graph=True)
            reward = em.take_action(action)
            print("steps: ")
            print(t)
            time_list[i] = t
            print("reward: ")
            print(reward.item())
            if em.done:
                break
        

    em.close()

    print("final sums of time")
    print(time_list)


    
def main():
    train()
    # test_trained_model()

if __name__ == "__main__":
    main()