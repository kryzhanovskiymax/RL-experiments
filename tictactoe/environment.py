import gym
from gym import spaces
from gym.utils import seeding
from enum import Enum

import numpy as np
import random


class TicTacToeEnv(gym.Env):
    environment_name = "TicTacToeEnv"

    def __init__(self,
                 rewards=dict(
                     pos_ep=1,
                     neg_ep=-1,
                     draw=0.5,
                     step=-0.1
                 ),
                 thresholds=dict(
                     win_rate=0.9,
                     draw_rate=0.1
                 ),
                 opponent_type='random'):
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(3),spaces.Discrete(3))
        )
        self.rewards = rewards
        self.thresholds = thresholds
        self.stats = {
            'games_played': 0
        }

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self._get_obs()

    def _get_obs(self):
        return self.board

    def _check_win(self, player):
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        if b[0, 0] == b[1, 1] == b[2, 2] == player or b[0, 2] == b[1, 1] == b[2, 0] == player:
            return True
        return False

    def _check_draw(self):
        return np.all(self.board != 0)

    def opponent_policy(self):
        available_moves = np.where(self.board == 0)[0]
        if self.opponent_type == 'random':
            return random.choice(available_moves)
        elif self.opponent_type == 'human':
            move = int(input(f"Enter your move (0-8): "))  # Simple CLI input for human
            while move not in available_moves:
                move = int(input(f"Invalid move. Enter your move (0-8): "))
            return move
        else:
            raise ValueError(f"Unknown opponent type: {self.opponent_type}")

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError("Invalid action. Cell already taken.")

        # Player 1 move
        self.board[action] = self.current_player

        if self._check_win(self.current_player):
            return self._get_obs(), self.rewards['pos_ep'], True, {}

        if self._check_draw():
            return self._get_obs(), self.rewards['draw'], True, {}

        # Opponent move
        opponent_action = self.opponent_policy()
        self.board[opponent_action] = 2  # Opponent is always 'O'

        if self._check_win(2):
            return self._get_obs(), self.rewards['neg_ep'], True, {}

        if self._check_draw():
            return self._get_obs(), self.rewards['draw'], True, {}

        return self._get_obs(), self.rewards['step'], False, {}
