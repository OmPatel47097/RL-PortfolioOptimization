import matplotlib.pyplot as plt
from portfolio_env import PortfolioEnv
from models.ppo_actor_critic import ActorCritic
import numpy as np

# Load environment
env = PortfolioEnv(data_file='../data/stockdata.csv')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize agent
agent = ActorCritic(state_size=state_size, action_size=action_size)

num_episodes = 10
max_steps = 100

# Metrics
returns = []
cumulative_returns = []
steps = []

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_return = 0
    for t in range(max_steps):
        state = np.reshape(state, [1, state_size])
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.train(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward
        if done:
            break
            # Log the actions and rewards
        print(f"Step {t}: Action {action}, Reward {reward}, Portfolio Value {env._calculate_reward()}, Cash {env.cash}")
    returns.append(episode_return)
    cumulative_return = sum(returns) / (episode + 1)
    cumulative_returns.append(cumulative_return)
    steps.append(t)
    print(f"Episode {episode+1}/{num_episodes}, Return: {episode_return}")

# Plotting metrics
plt.figure(figsize=(10, 5))
plt.plot(returns, label='Returns')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Returns per Episode')
plt.legend()
plt.savefig('../reports/returns_per_episode.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(cumulative_returns, label='Average Cumulative Return')
plt.xlabel('Episode')
plt.ylabel('Cumulative Return')
plt.title('Average Cumulative Return per Episode')
plt.legend()
plt.savefig('../reports/cumulative_return_per_episode.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(steps, label='Steps per Episode')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.savefig('../reports/steps_per_episode.png')
plt.show()