import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

"""
Actor-Critic model for training the agent in the PortfolioEnv environment using the A2C algorithm. This model 
The actor network predicts a probability distribution over actions.
An action is sampled from this distribution and executed in the environment.
The environment returns the next state and the reward associated with the action taken.
"""


class ActorCritic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size      # 3 * len(self.data.columns) - Buy, Sell, Hold for each ticker
        self.gamma = 0.90  # discount factor
        # TODO: Explore learing rate as lower values help model to explore more and higher make model stuck.
        self.learning_rate = 0.00001
        # self.epsilon = 1.0  # Initial exploration rate
        # self.epsilon_min = 0.01
        # self.epsilon_decay = 0.995
        self.actor, self.critic = self._build_actor_critic()    # Build actor and critic networks
        self.optimizer = Adam(learning_rate=self.learning_rate)


    """
    Build the actor and critic networks using the Keras functional API. 
    """
    def _build_actor_critic(self):
        state_input = Input(shape=(self.state_size,))

        # Actor Network
        dense1 = Dense(24, activation='relu')(state_input)
        dense2 = Dense(24, activation='relu')(dense1)
        action_output = Dense(self.action_size, activation='softmax')(dense2)
        actor = Model(inputs=state_input, outputs=action_output)

        # Critic Network
        dense1 = Dense(24, activation='relu')(state_input)
        dense2 = Dense(24, activation='relu')(dense1)
        value_output = Dense(1, activation='linear')(dense2)
        critic = Model(inputs=state_input, outputs=value_output)

        return actor, critic

    def act(self, state):
        prob = self.actor.predict(state)[0]
        # print(prob)
        action = np.random.choice(self.action_size, p=prob)
        return action

    def train(self, state, action, reward, next_state, done):
        state = np.reshape(state, [1, self.state_size])
        next_state = np.reshape(next_state, [1, self.state_size])

        # Critic update - Calculate the target value using the reward and the value of the next state
        target = reward
        if not done:    # If the episode is not over, calculate the target value using the reward and the value of the next state
            target += self.gamma * self.critic.predict(next_state)[0]
        target = np.reshape(target, [1, 1])
        with tf.GradientTape() as tape:     # Calculate the loss and update the critic network
            value = self.critic(state, training=True)
            critic_loss = tf.reduce_mean(tf.square(target - value))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # Actor update - Calculate the advantage and update the actor network
        with tf.GradientTape() as tape:
            prob = self.actor(state, training=True)
            action_prob = prob[0][action]
            advantage = target - value
            actor_loss = -tf.math.log(action_prob) * advantage
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # # Decay epsilon
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
