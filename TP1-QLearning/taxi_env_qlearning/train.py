import numpy as np
import matplotlib.pyplot as plt
from taxi_env import TaxiEnvCustom
import os   
from dotenv import load_dotenv

load_dotenv()

env = TaxiEnvCustom()
q_table = np.zeros((env.state_space, env.action_space))

alpha = 0.1
gamma = 0.6
epsilon = 0.1
episodes = 10000
all_rewards = []

for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    for _ in range(200):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, env.action_space)
        else:
            action = np.argmax(q_table[state])
        next_state, reward, done, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
        state = next_state
        total_reward += reward
        if done:
            break
    all_rewards.append(total_reward)

rolling_avg = [np.mean(all_rewards[i:i+100]) for i in range(0, len(all_rewards), 100)]
plt.plot(rolling_avg)
plt.xlabel("Bloques de 100 episodios")
plt.ylabel("Recompensa promedio")
plt.title("Convergencia del aprendizaje")
plt.grid(True)
plt.show()


base_path = os.getenv("BASE_PATH")
q_table_path = os.path.join(base_path, os.getenv("Q_TABLE_PATH"))


np.save(q_table_path, q_table)