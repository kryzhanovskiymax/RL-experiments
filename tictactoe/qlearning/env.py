import numpy as np
import pickle
from agent import Agent, HumanAgent
from config import BOARD_COLS, BOARD_ROWS
from tqdm import tqdm


class TicTacToeEnv:
    def __init__(self, player1, player2):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.player1 = player1
        self.player2 = player2
        self.is_end = False
        self.board_hash = None
        self.current_player_symbol = 1

    def get_hash(self):
        self.board_hash = str(self.board.reshape(BOARD_COLS * BOARD_ROWS))
        return self.board_hash

    def check_winner(self):
        for i in range(BOARD_ROWS):
            if sum(self.board[i, :]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.is_end = True
                return -1
        for i in range(BOARD_COLS):
            if sum(self.board[:, i]) == 3:
                self.is_end = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.is_end = True
                return -1
        diag_sum1 = sum(self.board[i, i] for i in range(BOARD_COLS))
        diag_sum2 = sum(self.board[i, BOARD_COLS - i - 1] for i in range(BOARD_COLS))
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.is_end = True
            return 1 if diag_sum1 == 3 or diag_sum2 == 3 else -1
        if not self.available_positions():
            self.is_end = True
            return 0
        self.is_end = False
        return None

    def available_positions(self):
        positions = [(i, j) for i in range(BOARD_ROWS) for j in range(BOARD_COLS) if self.board[i, j] == 0]
        return positions

    def update_state(self, position):
        self.board[position] = self.current_player_symbol
        self.current_player_symbol = -1 if self.current_player_symbol == 1 else 1

    def give_reward(self):
        result = self.check_winner()
        if result == 1:
            self.player1.feed_reward(1)
            self.player2.feed_reward(0)
        elif result == -1:
            self.player1.feed_reward(0)
            self.player2.feed_reward(1)
        else:
            self.player1.feed_reward(0.1)
            self.player2.feed_reward(0.5)

    def reset(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.board_hash = None
        self.is_end = False
        self.current_player_symbol = 1

    def play(self, rounds=100):
        for _ in tqdm(range(rounds)):
            while not self.is_end:
                positions = self.available_positions()
                action = self.player1.choose_action(positions, self.board, self.current_player_symbol)
                self.update_state(action)
                board_hash = self.get_hash()
                self.player1.add_state(board_hash)

                win = self.check_winner()
                if win is not None:
                    self.give_reward()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset()
                    break

                positions = self.available_positions()
                action = self.player2.choose_action(positions, self.board, self.current_player_symbol)
                self.update_state(action)
                board_hash = self.get_hash()
                self.player2.add_state(board_hash)

                win = self.check_winner()
                if win is not None:
                    self.give_reward()
                    self.player1.reset()
                    self.player2.reset()
                    self.reset()
                    break

    def play_with_human(self):
        while not self.is_end:
            positions = self.available_positions()
            action = self.player1.choose_action(positions, self.board, self.current_player_symbol)
            self.update_state(action)
            self.render_board()
            win = self.check_winner()
            if win is not None:
                print(f"{self.player1.name} wins!" if win == 1 else "Tie!")
                self.reset()
                break

            positions = self.available_positions()
            action = self.player2.choose_action(positions)
            self.update_state(action)
            self.render_board()
            win = self.check_winner()
            if win is not None:
                print(f"{self.player2.name} wins!" if win == -1 else "Tie!")
                self.reset()
                break

    def render_board(self):
        symbols = {0: ' ', 1: 'x', -1: 'o'}
        print('-------------')
        for row in self.board:
            print('| ' + ' | '.join(symbols[cell] for cell in row) + ' |')
            print('-------------')


if __name__ == "__main__":
    # Training
    player1 = Agent("player1")
    player2 = Agent("player2")

    state = State(player1, player2)
    print("Training...")
    state.play(50000)

    # Play with human
    player1 = Agent("computer", exp_rate=0)
    player1.load_policy("policy_p1")

    player2 = HumanAgent("human")

    state = State(player1, player2)
    state.play_with_human()