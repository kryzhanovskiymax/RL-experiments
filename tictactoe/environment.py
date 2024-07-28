import numpy as np
import gym
from gym import spaces

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0,
                                            high=2,
                                            shape=(3, 3),
                                            dtype=np.int8)
        self.reset()
        
    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board
    
    def step(self, action, player):
        if self.done:
            raise ValueError("Game is already finished!")
        x, y = divmod(action, 3)
        if self.board[x, y] != 0:
            return self.board, -10, True, {}
        self.board[x, y] = player
        self.done, self.winner = self.check_done()
        reward = self.get_reward(player)        
        return self.board, self.get_reward(player), self.done, {}
    
    def check_done(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3 or abs(sum(self.board[:, i])) == 3:
                return True, np.sign(sum(self.board[i, :]) or sum(self.board[:, i]))
        diag1 = sum(self.board[i, i] for i in range(3))
        diag2 = sum(self.board[i, 2-i] for i in range(3))
        if abs(diag1) == 3 or abs(diag2) == 3:
            return True, np.sign(diag1 or diag2)
        if not any(0 in row for row in self.board):
            return True, 0
        return False, None
    
    def get_reward(self, player):
        if self.done:
            if self.winner == player:
                return 2
            elif self.winner == 0:
                return 1
            else:
                return -1
        return 0

    def is_valid_action(self, action, state):
        row, col = divmod(action, 3)
        return state[row, col] == 0

    def possible_actions(self, state):
        return [action for action in range(self.action_space.n) \
                if self.is_valid_action(action, state)]

    def render(self):
        print(self.board)

    def check_winner(self):
        board = self.board.reshape(3, 3)

        # Check rows
        for row in board:
            if abs(sum(row)) == 3:
                return np.sign(sum(row))

        # Check columns
        for col in board.T:
            if abs(sum(col)) == 3:
                return np.sign(sum(col))

        # Check diagonals
        if abs(board.trace()) == 3:
            return np.sign(board.trace())
        if abs(np.fliplr(board).trace()) == 3:
            return np.sign(np.fliplr(board).trace())

        # No winner
        return 0
