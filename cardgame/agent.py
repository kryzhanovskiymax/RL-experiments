import numpy as np
from game import Action


class Agent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.99):
        # Initialize Q-table with zeros
        self.Q = np.zeros((13, 41, 2, 2))  # (cards, effective_stack, action, agent_role)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, card, effective_stack, is_small_blind):
        # Epsilon-greedy action selection
        if np.random.rand() < self.exploration_rate:
            return np.random.choice([Action.FOLD, Action.PUSH])
        else:
            role_index = 0 if is_small_blind else 1
            return Action.PUSH if np.argmax(self.Q[card, effective_stack, :, role_index]) == 1 else Action.FOLD

    def feed_reward(self, card, effective_stack, action, reward, is_small_blind):
        role_index = 0 if is_small_blind else 1
        current_state = self.Q[card, effective_stack - 10, :, role_index]
        action_index = 1 if action == Action.PUSH else 0
        next_state_max = np.max(current_state) 
        self.Q[card, effective_stack - 10, action_index, role_index] += self.learning_rate * \
                        (reward + self.discount_factor * next_state_max - current_state[action_index])
        self.exploration_rate *= self.exploration_decay
