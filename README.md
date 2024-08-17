Here's a detailed `README.md` file that you can use for your GitHub project:

---

# **Automated Portfolio Management Using Reinforcement Learning**

## **Project Overview**

This project aims to develop an automated portfolio management system leveraging reinforcement learning (RL) techniques. Specifically, we implement and compare two advanced RL algorithms: **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** with an Actor-Critic model. The objective is to optimize asset allocation in a portfolio to maximize returns while minimizing risk.

## **Key Features**

- **Data Collection:** Historical stock data from 2010 to 2024 using the `yfinance` library.
- **Custom Environment:** A simulated trading environment created using the OpenAI Gym library, including action and state spaces tailored to financial markets.
- **Reinforcement Learning Algorithms:** Implementation of DQN and PPO algorithms for decision-making in dynamic and uncertain market conditions.
- **Evaluation & Comparison:** Extensive experimentation with different hyperparameters and batch sizes, including in-depth analysis of algorithm performance.

## **Project Structure**

- **`data/`:** Directory containing the collected historical stock data (e.g., `stockdata.csv`).
- **`scripts/`:**
  - **`get_data.py`:** Script for downloading and saving historical stock data using `yfinance`.
  - **`portfolio_env.py`:** Custom environment script that simulates the trading environment using Gym.
  - **`train_dqn.py`:** Script for training the DQN agent.
  - **`train_ppo.py`:** Script for training the PPO agent with an Actor-Critic model.
- **`models/`:**
  - **`dqn.py`:** Implementation of the DQN algorithm.
  - **`ppo_actor_critic.py`:** Implementation of the PPO algorithm with an Actor-Critic model.
- **`reports/`:** Directory for storing generated plots and analysis reports.
- **`logs/`:** Directory for TensorBoard logs.
- **`README.md`:** This file, providing an overview of the project.
- **`requirements.txt`:** A list of Python packages required to run the project.

## **Getting Started**

### **Prerequisites**

Ensure you have Python 3.7+ installed on your system. Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

### **Setting Up the Project**

1. **Clone the Repository:**

   ```bash
   git clone [https://github.com/OmPatel47097/RL-PortfolioOptimization.git](https://github.com/OmPatel47097/RL-PortfolioOptimization.git)
   cd RL-PortfolioOptimization
   ```

2. **Download Stock Data:**

   Run the data collection script to fetch historical stock data:

   ```bash
   python scripts/get_data.py
   ```

   This will download data for the specified tickers and store it in the `data/` directory.

3. **Train the DQN Model:**

   To train the DQN agent, run:

   ```bash
   python scripts/train_dqn.py
   ```

4. **Train the PPO Model:**

   To train the PPO agent, run:

   ```bash
   python scripts/train_ppo.py
   ```

### **Running TensorBoard**

To monitor training progress and view metrics like episode returns and cumulative returns:

```bash
tensorboard --logdir ./logs/
```

Then open `http://localhost:6006/` in your web browser.

## **Results**

After training, the performance of both algorithms can be evaluated using the generated plots in the `reports/` directory. These include:

- **Returns per Episode**
- **Cumulative Return per Episode**
- **Return per Episode**

The results demonstrate the effectiveness of DQN and PPO in managing a portfolio, with detailed analysis provided in the project's final report.

## **Challenges & Learnings**

Some of the major challenges encountered during the project included:

- **Reward Calculation Issues:** Division by zero errors affected reward calculation, leading to NaN values and disrupting the learning process.
- **Computation Power:** The high computational demands of the project required significant GPU resources and extended training times.

## **Future Work**

Future improvements could include:

- **Environment Refinements:** Enhancing the environment's realism by incorporating more complex financial instruments and market conditions.
- **Algorithm Enhancements:** Exploring hybrid models or integrating additional RL algorithms like A2C or SAC.
- **Scalability:** Leveraging cloud-based resources or distributed training to handle the computational load more efficiently.

## **Contributors**

- **Om Patel, Ujas Patel, Balbhadra Prajapati**

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
