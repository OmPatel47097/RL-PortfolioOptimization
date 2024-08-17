import matplotlib.pyplot as plt
from portfolio_env import PortfolioEnv
from models.ppo_actor_critic import ActorCritic
from collections import deque
import numpy as np
import os
import tensorflow as tf

def run_ppo_experiment(params):
    # Load environment
    env = PortfolioEnv(data_file='../data/stockdata.csv')

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    log_dir = f"../logs/ppo_{params['learning_rate']}_{params['gamma']}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    agent = ActorCritic(state_size=state_size, action_size=action_size)
    agent.learning_rate = params['learning_rate']
    agent.gamma = params['gamma']

    num_episodes = 1000
    max_steps = 10

    returns = []
    cumulative_returns = []
    steps = []

    avg_rewards = deque(maxlen=100)
    writer = tf.summary.create_file_writer(log_dir)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_return = 0
        print(f"\nEpisode {episode+1}/{num_episodes}")
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.train(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            if done:
                break
            print(f"Step {t}: Action {action}, Reward {reward}, Portfolio Value {env.portfolio_value}, Cash {env.cash}")

        returns.append(episode_return)
        cumulative_return = sum(returns) / (episode + 1)
        cumulative_returns.append(cumulative_return)
        steps.append(t)

        avg_rewards.append(episode_return)

        with writer.as_default():
            tf.summary.scalar('Score', episode_return, step=episode)
            tf.summary.scalar('Avg Reward (last 100)', np.mean(avg_rewards), step=episode)
            tf.summary.scalar('Cumulative Return', cumulative_return, step=episode)
            # tf.summary.scalar('Epsilon', agent.epsilon, step=episode)

        print(f"Episode {episode+1} Return: {episode_return}, Cumulative Return: {cumulative_return}, Steps: {t}")

    plot_metrics(returns, cumulative_returns, steps, params)

def plot_metrics(returns, cumulative_returns, steps, params):
    plt.figure(figsize=(10, 5))
    plt.plot(returns, label='Returns - PPO')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Returns per Episode')
    plt.legend()
    plt.savefig(f'../reports/returns_per_episode_ppo_lr{params["learning_rate"]}_gamma{params["gamma"]}.png')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_returns, label='Average Cumulative Return - PPO')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Return')
    plt.title('Average Cumulative Return per Episode')
    plt.legend()
    plt.savefig(f'../reports/cumulative_return_per_episode_ppo_lr{params["learning_rate"]}_gamma{params["gamma"]}.png')
    plt.show()

    # no need to plot steps per episode as it is same as stated above for all.
    # plt.figure(figsize=(10, 5))
    # plt.plot(steps, label='Steps per Episode - PPO')
    # plt.xlabel('Episode')
    # plt.ylabel('Steps')
    # plt.title('Steps per Episode')
    # plt.legend()
    # plt.savefig(f'../reports/steps_per_episode_ppo_lr{params["learning_rate"]}_gamma{params["gamma"]}.png')
    # plt.show()

if __name__ == "__main__":
    hyperparameters = [
        {'learning_rate': 0.00001, 'gamma': 0.95},
        {'learning_rate': 0.00001, 'gamma': 0.8},
        {'learning_rate': 1e-5, 'gamma': 0.99},
        {'learning_rate': 0.1, 'gamma': 0.99}
    ]

    for params in hyperparameters:
        run_ppo_experiment(params)


