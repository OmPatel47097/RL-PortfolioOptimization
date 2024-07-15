import numpy as np
import matplotlib.pyplot as plt
from portfolio_env import PortfolioEnv
from models.dqn import DQNAgent

# Load environment
env = PortfolioEnv(data_file='../data/stockdata.csv')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize agent
agent = DQNAgent(state_size=state_size, action_size=action_size)

num_episodes = 1000
max_steps = 1000
batch_size = 32

# Metrics
returns = []
cumulative_returns = []
steps = []

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
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        if done:
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        # Log the actions and rewards
        print(f"Step {t}: Action {action}, Reward {reward}, Portfolio Value {env.portfolio_value}, Cash {env.cash}")
    returns.append(episode_return)
    cumulative_return = sum(returns) / (episode + 1)
    cumulative_returns.append(cumulative_return)
    steps.append(t)
    print(f"Episode {episode+1} Return: {episode_return}, Cumulative Return: {cumulative_return}, Steps: {t}")

# Plotting metrics
plt.figure(figsize=(10, 5))
plt.plot(returns, label='Returns - DQN')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Returns per Episode')
plt.legend()
plt.savefig('../reports/returns_per_episode_dqn.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label='Average Cumulative Return - DQN')
plt.xlabel('Episode')
plt.ylabel('Cumulative Return')
plt.title('Average Cumulative Return per Episode')
plt.legend()
plt.savefig('../reports/cumulative_return_per_episode_dqn.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(steps, label='Steps per Episode - DQN')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.savefig('../reports/steps_per_episode_dqn.png')
plt.show()
