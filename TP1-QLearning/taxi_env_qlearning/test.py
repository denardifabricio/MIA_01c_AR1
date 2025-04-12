import time
import numpy as np
from taxi_env import TaxiEnvCustom
import os
from dotenv import load_dotenv

load_dotenv()

base_path = os.getenv("BASE_PATH")
q_table_path = os.path.join(base_path, os.getenv("Q_TABLE_PATH"))


img_base_path = os.path.join(base_path, "img")
env = TaxiEnvCustom(img_base_path=img_base_path)


q_table = np.load(q_table_path)




state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.5)
