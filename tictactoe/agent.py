import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

class QAgentNetwork(torch.nn.Module):
    def __init__(self,
                 input_size=9,
                 output_size=9,
                 hidden_layer_dim=64,
                 num_hidden_layers=3):
        '''
            Class for DQNAgent network
        '''
        super(QAgentNetwork, self).__init__()
        self.activation = torch.nn.ReLU()
        self.input_layer = torch.nn.Linear(input_size, hidden_layer_dim)
        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Linear(hidden_layer_dim,
                                                      hidden_layer_dim))
        
        self.output_layer = torch.nn.Linear(hidden_layer_dim, output_size)
    
    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x


class HumanAgent:
    def __init__(self,
                 env,
                 name='HumanAgent'):
        self.env = env
        self.name = name
    
    def choose_action(self, state, mode='train'):
        """
        Prompt the human player to input their action.
        """
        possible_actions = self.env.possible_actions(state)
        print("Possible actions: ", possible_actions)
        action = None
        while action not in possible_actions:
            try:
                action = int(input(f"Choose your action from {possible_actions}: "))
                if action not in possible_actions:
                    print(f"Action {action} is not valid. Choose again.")
            except ValueError:
                print("Invalid input. Please enter an integer.")
        return action

    def remember(self, state, action, reward, next_state, done):
        """
        for compatibility
        """
        pass

    def replay(self):
        """
        for compatibility
        """
        pass
    
    def log_episode_rewards(self, episode, reward):
        """
        for compatibility.
        """
        pass
    
    def close_writer(self):
        """
        for compatibility.
        """
        pass



class RandomAgent:
    def __init__(self,
                 env,
                 name='RandomAgent'):
        self.env = env
        self.name = name
    
    def choose_action(self, state, mode='train'):
        """
        Choose a random action from the possible actions.
        """
        possible_actions = self.env.possible_actions(state)
        return random.choice(possible_actions)

    def remember(self, state, action, reward, next_state, done):
        """
        for compatibility
        """
        pass

    def replay(self):
        """
        for compatibility
        """
        pass
    
    def log_episode_rewards(self, episode, reward):
        """
        for compatibility.
        """
        pass
    
    def close_writer(self):
        """
        for compatibility.
        """
        pass


class FixedStrategyAgent:
    def __init__(self,
                 env,
                 name='FixedStrategyAgent'):
        self.env = env
        self.name = name
    
    def choose_action(self, state, mode='train'):
        """
        Choose the action with the minimum ID from the possible actions.
        """
        possible_actions = self.env.possible_actions(state)
        return min(possible_actions)

    def remember(self, state, action, reward, next_state, done):
        """
        for compatibility
        """
        pass

    def replay(self):
        """
        for compatibility
        """
        pass
    
    def log_episode_rewards(self, episode, reward):
        """
        for compatibility
        """
        pass
    
    def close_writer(self):
        """
        for compatibility
        """
        pass


class QAgent:
    def __init__(self,
                 env,
                 name='Agent',
                 state_size=9,
                 action_size=9,
                 network=QAgentNetwork(),
                 lr=0.001,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.99,
                 batch_size=64,
                 memory_size=10000,
                 update_rate=10,
                 log_dir=None):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_rate = update_rate
        self.model = network
        self.target_model = network
        self.update_target_network()
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.train_step = 0
        self.writer = SummaryWriter(log_dir) if log_dir else SummaryWriter()
        self.episode_rewards = []
        self.name = name

    def choose_action(self, state, mode='train'):
        possible_actions = self.env.possible_actions(state)
        
        if mode == 'train':
            if np.random.rand() <= self.epsilon:
                return random.choice(possible_actions)
            
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            with torch.no_grad():
                act_values = self.model(state).numpy().flatten()
            
            # Filter act_values to only keep values for possible actions
            filtered_act_values = np.full_like(act_values, -np.inf)
            filtered_act_values[possible_actions] = act_values[possible_actions]
            
            return np.argmax(filtered_act_values)
        
        elif mode == 'eval':
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            with torch.no_grad():
                act_values = self.model(state).numpy().flatten()
            filtered_act_values = np.full_like(act_values, -np.inf)
            filtered_act_values[possible_actions] = act_values[possible_actions]
            
            return np.argmax(filtered_act_values)
        
        else:
            raise ValueError("Mode should be either 'train' or 'eval'")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        total_loss = 0
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state).clone().detach()
            target_f[0][action] = target
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / self.batch_size
        self.writer.add_scalar(f'{self.name}/Loss/train',
                               avg_loss,
                               self.train_step)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.writer.add_scalar(f'{self.name}/Epsilon/train',
                               self.epsilon,
                               self.train_step)

        self.train_step += 1
        if self.train_step % self.update_rate == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def log_episode_rewards(self, episode, reward):
        self.episode_rewards.append(reward)
        self.writer.add_scalar(f'{self.name}/Reward/episode', reward, episode)

    def log_reward_n_smooth(self, n):
        if len(self.episode_rewards) < n:
            return
        
        mean_reward = np.mean(self.episode_rewards[-n:])
        self.writer.add_scalar(f'{self.name}/Reward/mean_last_{n}_episodes',
                               mean_reward,
                               len(self.episode_rewards))

    def close_writer(self):
        self.writer.close()

    def clear_replay_buffer(self):
        self.memory = deque(maxlen=self.memory.maxlen)

    def reset_model(self):
        self.model = QAgentNetwork()
        self.target_model = QAgentNetwork()
        self.update_target_network()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.lr)  
