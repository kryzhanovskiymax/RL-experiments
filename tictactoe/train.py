import numpy as np
from tqdm import tqdm
from environment import TicTacToeEnv

class DQNTrainer:
    def __init__(self,
                 env=TicTacToeEnv(),
                 agent1=None,
                 agent2=None):
        self.env = env
        self.agent1 = agent1 or QAgent(env)
        self.agent2 = agent2 or QAgent(env)

    def train(self, episodes=100, log_step=10, logging=True, board_render=False):
        for e in tqdm(range(episodes)):
            state = self.env.reset()
            done = False
            turn = 1  # Player 1 starts
            total_reward1 = 0
            total_reward2 = 0
            if logging:
                print(f"Episode {e + 1}/{episodes}")
            
            while not done:
                if turn == 1:
                    action = self.agent1.choose_action(state, mode='train')
                    next_state, reward, done, _ = self.env.step(action, 1)
                    self.agent1.remember(state, action, reward, next_state, done)
                    state = next_state
                    turn = -1
                    total_reward1 += reward
                    x, y = divmod(action, 3)
                    if logging:
                        print(f"Player 1, action [{x}, {y}], reward {reward}")
                else:
                    action = self.agent2.choose_action(state, mode='train')
                    next_state, reward, done, _ = self.env.step(action, -1)
                    self.agent2.remember(state, action, reward, next_state, done)
                    state = next_state
                    turn = 1
                    total_reward2 += reward
                    x, y = divmod(action, 3)
                    if logging:
                        print(f"Player 2, action [{x}, {y}], reward {reward}")
    
                self.agent1.replay()
                self.agent2.replay()
            if e % log_step == 0:
                self.agent1.log_episode_rewards(e, total_reward1)
                self.agent2.log_episode_rewards(e, total_reward2)
            
            if board_render:
                print(f"Board after game {e+1}:")
                self.env.render()
        
        self.agent1.close_writer()
        self.agent2.close_writer()

    def eval(self,
             n_games=10,
             render_each_game=False,
             render_final_board=False):
        results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}

        for game in range(n_games):
            state = self.env.reset()
            done = False
            turn = 1  # Player 1 starts
            
            while not done:
                if turn == 1:
                    action = self.agent1.choose_action(state, mode='eval')
                    next_state, reward, done, _ = self.env.step(action, 1)
                    state = next_state
                    turn = -1
                else:
                    action = self.agent2.choose_action(state, mode='eval')
                    next_state, reward, done, _ = self.env.step(action, -1)
                    state = next_state
                    turn = 1
                
                if render_each_game:
                    self.env.render()  # Display the current state of the environment
            
            # Log results based on the final state of the game
            winner = self.env.check_winner()
            if winner == 1:
                results['agent1_wins'] += 1
            elif winner == -1:
                results['agent2_wins'] += 1
            else:
                results['draws'] += 1

            if render_final_board:
                print(f"Final board of game {game + 1}:")
                self.env.render()
                print(f"Game {game + 1} finished. Winner: {'Agent 1' if winner == 1 else 'Agent 2' if winner == -1 else 'Draw'}")
        
        # Print final evaluation results
        print(f"Evaluation over {n_games} games:")
        print(f"{self.agent1.name} wins: {results['agent1_wins']} ({results['agent1_wins'] / n_games:.2f})")
        print(f"{self.agent2.name} wins: {results['agent2_wins']} ({results['agent2_wins'] / n_games:.2f})")
        print(f"Draws: {results['draws']} ({results['draws'] / n_games:.2f})")

        # Close the environment's rendering window if needed (depends on your environment implementation)
        if hasattr(self.env, 'close'):
            self.env.close()


def game(player1, player2, env=TicTacToeEnv(), log_board=False):
    '''
    Performs a game between two players.
    Args:
    - player1: player 1 that performs actions
    - player2: player 2 that performs actions
    - env: the TicTacToe environment
    - log_board: whether to log the board after each action or not
    Returns:
    - winner: 1 if player1 wins, -1 if player2 wins, 0 if draw
    '''
    state = env.reset()
    done = False
    turn = 1
    
    while not done:
        if turn == 1:
            action = player1.choose_action(state, mode='eval')
            next_state, reward, done, _ = env.step(action, 1)
            state = next_state
            turn = -1
        else:
            action = player2.choose_action(state, mode='eval')
            next_state, reward, done, _ = env.step(action, -1)
            state = next_state
            turn = 1

        if log_board:
            env.render()
    
    winner = env.check_winner()
    if log_board:
        print(f"Game finished. Winner: {player1.name if winner == 1 else player2.name if winner == -1 else 'Draw'}")
    
    return winner
