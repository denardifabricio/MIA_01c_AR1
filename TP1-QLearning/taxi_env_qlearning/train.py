import optuna
import numpy as np
import matplotlib.pyplot as plt
from taxi_env import TaxiEnvCustom
import os
from dotenv import load_dotenv
import csv

# Cargar variables de entorno desde .env
load_dotenv()
base_path = os.getenv("BASE_PATH")
q_table_path = os.path.join(base_path, "best_q_table.npy")

# Variable global para rastrear la mejor recompensa promedio
best_avg_reward = -float("inf")  # Inicializar con un valor muy bajo


def train_q_learning(alpha, gamma, epsilon, episodes=1000):
    """Entrena un agente Q-Learning con los hiperparámetros dados."""
    env = TaxiEnvCustom()
    q_table = np.zeros((env.state_space, env.action_space))
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(200):  # Limitar a 200 pasos por episodio
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, env.action_space)  # Acción aleatoria
            else:
                action = np.argmax(q_table[state])  # Mejor acción según la Q-table

            next_state, reward, done, _ = env.step(action)
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            # Actualizar Q-value
            q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)

    # Calcular la recompensa promedio
    avg_reward = np.mean(total_rewards[-100:])  # Promedio de los últimos 100 episodios
    return avg_reward, q_table


def objective(trial):
    """Función objetivo para Optuna."""
    global best_avg_reward  # Usar la variable global para rastrear la mejor recompensa promedio

    # Sugerir valores para los hiperparámetros
    alpha = trial.suggest_float("alpha", 0.1, 1.0, log=True)
    gamma = trial.suggest_float("gamma", 0.7, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)

    # Entrenar el agente con los hiperparámetros sugeridos
    avg_reward, q_table = train_q_learning(alpha, gamma, epsilon, episodes=1000)

    # Guardar la mejor Q-table si es la mejor recompensa promedio hasta ahora
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        np.save(q_table_path, q_table)

    return avg_reward


results_path = os.path.join(base_path,"results")
best_result_path = os.path.join(results_path,"best_result")

if __name__ == "__main__":
    # Crear un estudio de Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Probar 50 combinaciones

    # Mostrar los mejores hiperparámetros
    print("Mejores hiperparámetros:")
    print(study.best_params)
    print("Mejor recompensa promedio:")
    print(study.best_value)

    # Graficar la convergencia
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.show()

    # Mostrar una tabla con los resultados de cada trial
    print("\nResultados de cada trial:")
    print(f"{'Trial':<10}{'Alpha':<10}{'Gamma':<10}{'Epsilon':<10}{'Recompensa promedio':<20}")
    for trial in study.trials:
        # Guardar los resultados de cada trial en un archivo CSV

        csv_path = os.path.join(results_path, "trial_results.csv")
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Trial", "Alpha", "Gamma", "Epsilon", "Recompensa promedio", "Mejor"])
            for trial in study.trials:
                is_best = "Yes" if trial.number == study.best_trial.number else "No"
                writer.writerow([trial.number, trial.params['alpha'], trial.params['gamma'], trial.params['epsilon'], trial.value, is_best])
             # Guardar los resultados en una imagen
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        
        optimization_history_path = os.path.join(results_path, "optimization_history.png")

        # Guardar el gráfico de la importancia de los hiperparámetros
        importance_fig = optuna.visualization.matplotlib.plot_param_importances(study)
        param_importance_path = os.path.join(results_path, "param_importance.png")
        importance_fig.figure.savefig(param_importance_path)

        # Guardar el gráfico de convergencia
        fig.figure.savefig(optimization_history_path)

        # Graficar y guardar la relación entre los parámetros y la recompensa promedio
        for param in study.best_params.keys():
            slice_fig = optuna.visualization.matplotlib.plot_slice(study, params=[param])
            slice_path = os.path.join(results_path, f"slice_{param}.png")
            slice_fig.figure.savefig(slice_path)

 