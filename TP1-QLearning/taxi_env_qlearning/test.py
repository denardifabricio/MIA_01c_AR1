import time
import numpy as np
from taxi_env import TaxiEnvCustom
import os
env = TaxiEnvCustom()
base_path = "/Users/fabricio.denardi/Documents/CEIA/AR1/repos/MIA_01c_AR1/TP1-QLearning/taxi_env_qlearning/"

q_table_path = os.path.join(base_path, "q_table.npy")

q_table = np.load(q_table_path)




state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.5)
