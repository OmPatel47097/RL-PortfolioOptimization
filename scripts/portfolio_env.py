import gym
from gym import spaces
import numpy as np
import pandas as pd

INITIAL_CASH = 10000


class PortfolioEnv(gym.Env):
    def __init__(self, data_file, invest_cash=INITIAL_CASH):
        super(PortfolioEnv, self).__init__()
        self.data = pd.read_csv(data_file, index_col=0, header=[0, 1])
        self.data = self.data['Adj Close']  # Select only the 'Adj Close' column
        self.initial_cash = invest_cash
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold for each ticker
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.data.columns),), dtype=np.float32)
        self.reset()

    """
    Reset the environment to an initial state and return the initial observation
    """

    def reset(self):  # Reset the state of the environment to an initial state
        self.current_step = 0
        self.cash = self.initial_cash
        self.portfolio_value = self.cash
        self.holdings = {ticker: 0 for ticker in self.data.columns}
        return self._get_state()

    """
    Get the observation from the environment at the current time step
    """

    def _get_state(self):
        return np.array(self.data.iloc[self.current_step].values, dtype=np.float32)

    """
    Execute one time step within the environment and return the state, reward, done, and info
    """

    def step(self, action):
        prices = self.data.iloc[self.current_step].values
        self._execute_action(action, prices)
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = self.portfolio_value - self.initial_cash
        state = self._get_state()
        return state, reward, done, {}

    def _execute_action(self, action, prices):  # Buy, Sell, Hold
        for i, ticker in enumerate(self.data.columns):
            if action == 0:  # Hold
                continue
            elif action == 1:  # Buy
                quantity = self.cash // prices[i]
                self.cash -= quantity * prices[i]
                self.holdings[ticker] += quantity
            elif action == 2:  # Sell
                self.cash += self.holdings[ticker] * prices[i]
                self.holdings[ticker] = 0
        # Calculate portfolio value
        self.portfolio_value = self.cash + sum(
            self.holdings[ticker] * prices[i] for i, ticker in enumerate(self.data.columns))

    def _calculate_reward(self):
        return self.portfolio_value - self.initial_cash
