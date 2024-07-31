import numpy as np
from tqdm import tqdm
from environment import TicTacToeEnv

def train(agent, env, episodes=5000, log=True):

    for episode in episodes:
        env.reset()
        done = False
        state = env._get_obs()

        while not done:
            action = agent.choose_action(state, mode='train')
            next_state, reward, done, _ = env.step(action)
            next_state = np.eye(3)[next_state].flatten()  # One-hot encoding for the next state
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()

        agent.log_episode_rewards(episode, total_reward)
        agent.log_reward_n_smooth(10)
        if log:
            print(f"Episode {episode + 1}/{num_episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")
    agent.close_writer()
            
