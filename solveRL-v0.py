import gym
import math
import random
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

# env = gym.make('gym_auv:auv-v0')
# env.init_env(Motion_plan_state(x = 740.0, y = 280.0, z = -5.0, theta = 0), Motion_plan_state(x = 750.0, y = 280, z = -5.0, theta = 0), [])

# nn.Module base class for all neural network modules
class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        # 2 fully connected hidden layers
        # first layer will have 8 inputs
        #   auv 3D position and theta + shark 3D postion and theta
        self.fc1 = nn.Linear(in_features=8, out_features=24)  
        # TODO: for now, this should be ok? 
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        # only available output: 2 actions (v, w)
        self.out = nn.Linear(in_features=32, out_features=2)


    def forward(self, t):
        """
        define the forward pass through the neural network

        Parameters:
            t - the state
        """
        # Note, the state is a tuple of two np.array
        # we will convert the state to a flat tensor here:
        auv_tensor = torch.from_numpy(t[0])
        print(auv_tensor)
        shark_tensor = torch.from_numpy(t[1])
        # join 2 tensor together
        t = torch.cat((auv_tensor, shark_tensor)).float()
        print("processed t: ")
        print(t)
        # pass through the layers then have relu applied to it
        # relu is the activation function that will turn any negative value to 0,
        #   and keep any positive value
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)

        # pass through the last layer, the output layer
        t = self.out(t)

        print("from the output layer: ")
        print(t)
        print("-----")
        return t


# Create the experience class
# namedtuple allows you to create tuples with named fields, kind of like struct?
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        """
        function to store experience
        """
        # if there's space in the replay memory
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            # overwrite the oldest memory
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


"""
For implementing epsilon greedy strategy in choosing an action
(exploration vs exploitation)
"""
class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)


class Agent():
    def __init__(self, strategy, actions_range, device):
        # the agent's current step in the environment
        self.current_step = 0
        # in our case, the strategy will be the epsilon greedy strategy
        self.strategy = strategy
        # how many possible actions can the agent take at a given state
        # for our case, it only has 2 actions (left or right)
        self.actions_range = actions_range
        # what we want to PyTorch to use for tensor calculation
        self.device = device

        self.ra = None
        self.ea = None

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            v_range = self.actions_range[0]
            w_range = self.actions_range[1]
            v_action = np.random.uniform(v_range[0], v_range[1])
            w_action = np.random.uniform(w_range[0], w_range[1])
            print("-----")
            print("randomly chosen action: ")
            print(torch.tensor([v_action, w_action]))
            
            return torch.tensor([v_action, w_action]).to(self.device) # explore  
        else:
            # turn off gradient tracking bc we are using the model for inference instead of training
            # we don't need to keep track the gradient because we are not doing backpropagation to figure out the weight 
            # of each node yet
            with torch.no_grad():
                # for the given "state"，the output will be the action 
                #   with the highest Q-Value output from the policy net
                print("-----")
                print("exploiting")
                return policy_net(state).argmax(dim=0).to(self.device) # exploit


class AuvEnvManager():
    def __init__(self, device):
        self.device = device
        # have access to behind-the-scenes dynamics of the environment 
        self.env = gym.make('gym_auv:auv-v0').unwrapped
        self.env.init_env(Motion_plan_state(x = 740.0, y = 280.0, z = -5.0, theta = 0), Motion_plan_state(x = 750.0, y = 280, z = -5.0, theta = 0), [])

        self.current_state = None
        self.done = False

    def reset(self):
        self.env.reset()
        
    def close(self):
        self.env.close()
    
    def render(self, mode='human'):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        v_action = action[0].item()
        w_action = action[1].item()
        print("=========================")
        print("action v: ", v_action)  
        print("action w: ", w_action)    
        # we only care about the reward and whether or not the episode has ended
        # action is a tensor, so item() returns the value of a tensor (which is just a number)
        self.current_state, reward, self.done, _ = self.env.step((v_action, w_action))
        print("new state: ")
        print(self.current_state)
        print("reward: ")
        print(reward)
        print("=========================")
        # wrap reward into a tensor, so we have input and output to both be tensor
        return torch.tensor([reward], device=self.device)

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

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)


class QValues():
    """
    mainly use its static method
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        # returns the predicted q-values from the policy_net for the specific state-action pairs that were passed in.
        return policy_net(states).gather(dim=0, index=actions.unsqueeze(-1))

    @staticmethod        
    def get_next(target_net, next_states):  
        # for each next state, we want to obtain the max q-value predicted by the target_net among all the possible next actions              
        # we want to know where the final states are bc we shouldn't pass them into the target net
        # check individual next state's max value
        # if max value is 0 (.eq(0)), set to be true
        final_state_locations = next_states.flatten(start_dim=0) \
            .max(dim=0)[0].eq(0).type(torch.bool)
        # flip the non final_state 
        non_final_state_locations = (final_state_locations == False)

        non_final_states = next_states[non_final_state_locations]

        batch_size = next_states.shape[0]
        # create a tensor of zero
        values = torch.zeros(batch_size).to(QValues.device)

        # a tensor of 
        #   zero - if it's a final state
        #   target_net's maximum predicted q-value - if it's a non-final state.
        values[non_final_state_locations] = target_net(non_final_states).max(dim=0)[0].detach()
        return values


def main():
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

    num_episodes = 1000

    # use GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = AuvEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)

    agent = Agent(strategy, em.env.actions_range(), device)
    memory = ReplayMemory(memory_size)

    # to(device) puts the network on our defined device
    policy_net = DQN().to(device)
    target_net = DQN().to(device)

    # set the weight and bias in the target_net to be the same as the policy_net
    target_net.load_state_dict(policy_net.state_dict())
    # set the target_net in evaluation mode instead of training mode (bc we are only using it to 
    # estimate the next max Q value)
    target_net.eval()

    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

    episode_durations = []

    # For each episode:
    for episode in range(num_episodes):
        # Initialize the starting state.
        em.reset()
        state = em.get_state()
        
        for timestep in count(): 
            # For each time step:
            # Select an action (Via exploration or exploitation)
            action = agent.select_action(state, policy_net)
            print(action)
            print(type(action))
            # Execute selected action in an emulator.
            # Observe reward and next state.
            reward = em.take_action(action)
            next_state = em.get_state()

            # Store experience in replay memory.
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                # Sample random batch from replay memory.
                experiences = memory.sample(batch_size)
                # extract states, actions, rewards, next_states into their own individual tensors from experiences batch
                # 
                states, actions, rewards, next_states = extract_tensors(experiences)
                
                # Pass batch of preprocessed states to policy network.
                # return the q value for the given state-action pair by passing throught the policy net
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                # Calculate loss between output Q-values and target Q-values.
                # mse_loss calculate the mean square error
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))

                # Gradient descent updates weights in the policy network to minimize loss.
                # sets the gradients of all the weights and biases in the policy network to zero
                # so that we can do back propagation 
                optimizer.zero_grad()

                # use backward propagation to calculate the gradient of loss with respect to all the weights and biases in the policy net
                loss.backward()

                # updates the weights and biases of all the nodes based on the gradient
                optimizer.step()
            
            if em.done: 
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break

        #  After x time steps, weights in the target network are updated to the weights in the policy network.
        # in our case, it will be 10 episodes
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    em.close()


if __name__ == "__main__":
    main()