import numpy as np
from tqdm import tqdm
from environment import TicTacToeEnv
from tqdm import tqdm

def log_board(board):
    symbols = {0: ' ', 1: 'X', 2: 'O'}
    display_board = np.vectorize(symbols.get)(board)
    board_str = "\n".join(["|".join(row) for row in display_board])
    board_str = board_str.replace(' ', '_')
    board_str = board_str.replace('\n', '\n-----\n')
    
    print(board_str)

def train(agent,
          env,
          episodes=5000,
          log=True,
          log_episode=False,
          reward_smoothing=10,
          history=False):
    history_log = {
        'reward_smoothed': [],
        'reward': []
    }
    
    iterable = []
    if log:
        iterable = range(episodes)
    else:
        iterable = tqdm(range(episodes))
    for episode in iterable:
        if log_episode:
            print(f"Episode {episode}/{episodes}")
        env.reset()
        done = False
        state = env.board
        total_reward = 0
        while not done:
            if log_episode:
                log_board(state)
            action = agent.choose_action(state, mode='train')
            if log_episode:
                print(f"Action taken: {action}")
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay(log=True)

        agent.log_episode_rewards(episode, total_reward)
        mean_reward = agent.log_reward_n_smooth(reward_smoothing)

        if history:
            history_log['reward'].append(total_reward)
            if episode >= reward_smothing:
                history_log['reward_smoothed'].append(mean_reward)
        if log:
            print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")
            
