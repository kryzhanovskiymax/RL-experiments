import gym
from gym import spaces
import numpy as np
import random

class TicTacToeEnv(gym.Env):
    def __init__(self, opponent_type='random', agent=None):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0,
                                            high=2,
                                            shape=(3, 3),
                                            dtype=np.int8)
        self.opponent_type = opponent_type
        self.agent = agent
        self.seed()
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1 
        return self.board, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if self.board.flat[action] != 0:
            return self.board, -10, True, {}
        self.board.flat[action] = self.current_player

        if self._check_win(self.current_player):
            return self.board, 1 if self.current_player == 1 else -1, True, {}

        if self._check_draw():
            return self.board, 0.5, True, {}

        self.current_player = 2
        opponent_action = self.opponent_policy()
        self.board.flat[opponent_action] = self.current_player

        if self._check_win(self.current_player):
            return self.board, -1 if self.current_player == 2 else 1, True, {}

        if self._check_draw():
            return self.board, 0.5, True, {}

        self.current_player = 1
        return self.board, -0.1, False, {}

    def opponent_policy(self):
        if self.opponent_type == 'random':
            available_moves = self.possible_actions(self.board)
            return random.choice(available_moves)
        elif self.oppenent_type == 'agent':
            if self.agent is None:
                raise ValueError("Agent is not set")
            action = agent.choose_action(self.board, mode='eval')
            return action
        elif opponent_type == 'human':
            self.render()
            print('Make your turn')
        else:
            raise ValueError('Wrong oppenent type')

    def possible_actions(self, board):
        return [i for i, x in enumerate(board.flat) if x == 0]

    def render(self):
        '''
        Renders board
        '''
        symbols = {0: ' ', 1: 'X', 2: 'O'}
        board_str = "\n"
        for row in self.board:
            row_str = "|".join([symbols[cell] for cell in row])
            board_str += row_str + "\n" + "-" * 5 + "\n"
        print(board_str.strip("-\n")) 

    def _check_win(self, player):
        b = self.board
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        if b[0, 0] == b[1, 1] == b[2, 2] == player or b[0, 2] == b[1, 1] == b[2, 0] == player:
            return True
        return False

    def _check_draw(self):
        return np.all(self.board != 0)

# Register the environment
gym.envs.registration.register(
    id='TicTacToe-v0',
    entry_point=TicTacToeEnv,
)
