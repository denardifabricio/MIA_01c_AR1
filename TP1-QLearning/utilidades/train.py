import numpy as np

def train_q_learning(alpha, gamma, epsilon, env, episodes=1000, graficar_aprendizaje=False):
    """Entrena un agente Q-Learning con los hiperparámetros dados."""
    q_table = np.zeros((env.state_space, env.action_space))
    total_rewards = []
    steps_per_episode = []
    success_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        success = 0

        for _ in range(200):  # Limitar a 200 pasos por episodio
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
            steps += 1

            if done:
                success = 1
                break

        total_rewards.append(total_reward)
        steps_per_episode.append(steps)
        success_per_episode.append(success)

    avg_reward = np.mean(total_rewards[-100:])

    if graficar_aprendizaje:
        return avg_reward, q_table, total_rewards, steps_per_episode, success_per_episode
    else:
        return avg_reward, q_table