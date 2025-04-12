import time
import numpy as np
from taxi_env import TaxiEnvCustom
import os
from dotenv import load_dotenv

def save_step(save_step_path,num):
    env.save_render(img_path=os.path.join(save_step_path, f"step_{num:02}.png"))


def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


if __name__ == "__main__":
    load_dotenv()

    base_path = os.getenv("BASE_PATH")
    q_table_path = os.path.join(base_path, os.getenv("Q_TABLE_PATH"))


    img_base_path = os.path.join(base_path, "img")
    env = TaxiEnvCustom(img_base_path=img_base_path)

   

    q_table = np.load(q_table_path)

    state = env.reset()
    done = False
    env.render()


    step_number = 0
    save_step_path = os.path.join(base_path,"results","best_result","img")
    clear_folder(save_step_path)

    save_step(save_step_path,step_number)

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)

        img_path = os.path.join(img_base_path, f"step_{env.step}.png")
        

        save_step(save_step_path,step_number)

        step_number += 1
