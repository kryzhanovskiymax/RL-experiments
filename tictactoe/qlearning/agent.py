import numpy as np
import pickle
from config import BOARD_COLS, BOARD_ROWS


class Agent:
    def __init__(self,
                 name,
                 exp_rate=0.3,
                 lr=0.2,
                 decay_gamma=0.9):
        self.name = name
        self.states = [] 
        self.lr = lr
        self.exp_rate = exp_rate
        self.decay_gamma = decay_gamma
        self.states_value = {}

    def get_hash(self, board):
        board_hash = str(board.reshape(BOARD_COLS * BOARD_ROWS))
        return board_hash

    def choose_action(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            # Take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            action = None
            for pos in positions:
                next_board = current_board.copy()
                next_board[pos] = symbol
                next_board_hash = self.get_hash(next_board)
                value = self.states_value.get(next_board_hash, 0)
                if value >= value_max:
                    value_max = value
                    action = pos
        return action

    def add_state(self, state):
        self.states.append(state)

    def feed_reward(self, reward):
        for state in reversed(self.states):
            if state not in self.states_value:
                self.states_value[state] = 0
            self.states_value[state] += self.lr * (self.decay_gamma * reward - self.states_value[state])
            reward = self.states_value[state]

    def reset(self):
        self.states = []

    def save_policy(self):
        with open(f'policy_{self.name}', 'wb') as fw:
            pickle.dump(self.states_value, fw)

    def load_policy(self, file):
        with open(file, 'rb') as fr:
            self.states_value = pickle.load(fr)


class HumanAgent:
    def __init__(self, name):
        self.name = name

    def choose_action(self, positions):
        while True:
            row = int(input("Row: "))
            col = int(input("Col: "))
            action = (row, col)
            if action in positions:
                return action

    def add_state(self, state):
        pass

    def feed_reward(self, reward):
        pass

    def reset(self):
        pass
