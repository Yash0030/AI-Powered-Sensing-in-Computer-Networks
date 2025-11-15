import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import zmq
import json

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Simulation Parameters (Must match C++ side) ---
STATE_DIM = 5  # [edge_load, cloud_load,energy_level, task_size, task_compute]
ACTION_DIM = 3 # 0: Process locally, 1: Offload to Edge, 2: Offload to Cloud
BATCH_SIZE = 64
EPISODES = 30 # Train for 50 episodes
MODEL_FILE_PATH = "ppo_model_weights" # File path to save/load weights

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, clip_ratio=0.2, epochs=10):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.clear_memory()

    def _build_actor(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        return tf.keras.Model(inputs, outputs)

    def _build_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1)(x)
        return tf.keras.Model(inputs, outputs)

    def choose_action(self, state):
        """For TRAINING: Uses stochastic policy (exploration)"""
        state = np.reshape(state, [1, self.state_dim])
        probs = self.actor(state)
        action = tf.random.categorical(tf.math.log(probs), 1).numpy().flatten()[0]
        log_prob = tf.math.log(probs[0, action])
        return action, log_prob.numpy() # Return numpy value

    def choose_action_deterministic(self, state):
        """For TESTING: Uses deterministic policy (picks best action)"""
        state = np.reshape(state, [1, self.state_dim])
        probs = self.actor(state).numpy()
        action = np.argmax(probs) # Choose the action with the highest probability
        return action

    def get_action_probabilities(self, state):
        """For TESTING: Returns the raw probabilities for AUC calculation"""
        state = np.reshape(state, [1, self.state_dim])
        probs = self.actor(state).numpy().flatten()
        return probs

    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []

    def learn(self):
        states = tf.convert_to_tensor(self.states, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(self.rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(self.next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(self.dones, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(self.log_probs, dtype=tf.float32)

        values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = self._calculate_advantages(rewards, values, next_values, dones)
        
        # Calculate returns (target for critic)
        returns = advantages + values.numpy().flatten()
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)

        for _ in range(self.epochs):
            with tf.GradientTape() as tape:
                new_probs = self.actor(states)
                new_log_probs = tf.math.log(tf.gather_nd(new_probs, tf.expand_dims(actions, axis=1), batch_dims=1))
                
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

                # Critic loss: Compare current value to GAE-based returns
                current_values = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(returns - tf.squeeze(current_values)))
                
                total_loss = actor_loss + 0.5 * critic_loss

            grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

        self.clear_memory()

    def _calculate_advantages(self, rewards, values, next_values, dones):
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0
        
        values = values.numpy().flatten()
        next_values = next_values.numpy().flatten()
        rewards = rewards.numpy().flatten()
        dones = dones.numpy().flatten()

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[t]) # 0.95 is GAE lambda
            advantages[t] = gae
        return advantages
    
    def save_weights(self, filepath):
        """Saves the actor and critic model weights"""
        print(f"Saving weights to {filepath}...")
        self.actor.save_weights(filepath + '_actor.weights.h5')
        self.critic.save_weights(filepath + '_critic.weights.h5')

    def load_weights(self, filepath):
        """Loads the actor and critic model weights"""
        print(f"Loading weights from {filepath}...")
        self.actor.load_weights(filepath + '_actor.weights.h5')
        self.critic.load_weights(filepath + '_critic.weights.h5')

def main():
    print("Starting Python DRL Client...")
    
    # 1. Setup ZMQ Client
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")
    print("Connected to C++ NS-3 Server.")

    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    
    for i in range(EPISODES):
        # 2. Get initial state from C++ server
        socket.send_json({"command": "reset"})
        state_msg = socket.recv_json()
        state = np.array(state_msg["state"])
        
        done = False
        total_reward = 0
        
        while not done:
            # 3. Choose action in Python (with exploration)
            action, log_prob = agent.choose_action(state)
            
            # 4. Send action to C++, get result
            socket.send_json({"command": "step", "action": int(action)})
            result_msg = socket.recv_json()
            
            next_state = np.array(result_msg["next_state"])
            reward = result_msg["reward"]
            done = result_msg["done"]
            
            # 5. Store transition and learn
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            state = next_state
            total_reward += reward

            if len(agent.states) >= BATCH_SIZE:
                agent.learn()
                
        print(f"Episode: {i+1}, Total Reward: {total_reward:.2f}, Final Energy: {(state[2]*100):.2f}%")

    print("Training Finished.")
    
    # --- MODIFICATION: SAVE THE TRAINED WEIGHTS ---
    agent.save_weights(MODEL_FILE_PATH)
    
    socket.send_json({"command": "shutdown"})
    socket.recv_json() # Wait for ack

if __name__ == '__main__':
    main()
