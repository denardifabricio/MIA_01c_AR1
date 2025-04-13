import optuna
import numpy as np
import os
import csv
from utilidades.taxi_env import TaxiEnvCustom, remove_obstacles, generate_city_blocks
from utilidades.train import train_q_learning
from utilidades.graficar import plot_learning_curve, plot_study_results
from utilidades.evaluate import evaluate_q_table

base_path = "./"
q_table_path = os.path.join(base_path, "results/best_q_table.npy")
results_path = os.path.join(base_path,"results")
best_result_path = os.path.join(results_path,"best_result")

GRID_SIZE = 10
PICKUP_LOCATIONS = [(1, 1), (8, 7), (4, 2), (2, 8)]
DROPOFF_LOCATIONS = [(5, 5), (5, 4), (4, 5), (4, 4)]
OBSTACLES = remove_obstacles(generate_city_blocks(GRID_SIZE), PICKUP_LOCATIONS+DROPOFF_LOCATIONS)

taxi_env = TaxiEnvCustom(GRID_SIZE, PICKUP_LOCATIONS, DROPOFF_LOCATIONS, OBSTACLES)

# Variable global para rastrear la mejor recompensa promedio
best_avg_reward = -float("inf")  # Inicializar con un valor muy bajo

def objective(trial):
    """Función objetivo para Optuna."""
    global best_avg_reward  # Usar la variable global para rastrear la mejor recompensa promedio

    # Sugerir valores para los hiperparámetros
    alpha = trial.suggest_float("alpha", 0.1, 0.5, log=True)
    gamma = trial.suggest_float("gamma", 0.7, 0.99)
    epsilon = trial.suggest_float("epsilon", 0.1, 1.0)

    # Entrenar el agente con los hiperparámetros sugeridos
    avg_reward, q_table = train_q_learning(alpha, gamma, epsilon, env=taxi_env, episodes=2000)

    # Guardar la mejor Q-table si es la mejor recompensa promedio hasta ahora
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        np.save(q_table_path, q_table)

    return avg_reward

if __name__ == "__main__":
    # Crear un estudio de Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)  # Probar 50 combinaciones

    # Mostrar los mejores hiperparámetros
    print("Mejores hiperparámetros:")
    print(study.best_params)
    print("Mejor recompensa promedio:")
    print(study.best_value)

    # Mostrar una tabla con los resultados de cada trial
    print("\nGuardar resultados de cada trial:")
    csv_path = os.path.join(results_path, "trial_results.csv")
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Trial", "Alpha", "Gamma", "Epsilon", "Recompensa promedio", "Mejor"])
        for trial in study.trials:
            is_best = "Yes" if trial.number == study.best_trial.number else "No"
            writer.writerow([
                trial.number,
                trial.params['alpha'],
                trial.params['gamma'],
                trial.params['epsilon'],
                trial.value,
                is_best
            ])
    
    # Guardar gráficos de estudio
    plot_study_results(study, results_path)

    print("iniciando entrenamiento final")
    
    final_avg_reward, final_q_table, final_rewards, final_steps, final_success = train_q_learning(
        alpha=study.best_params['alpha'],
        gamma=study.best_params['gamma'],
        epsilon=study.best_params['epsilon'],
        env=taxi_env,
        episodes=3000, # o 50_000 si el entorno es grande
        graficar_aprendizaje=True
    )
    
    plot_learning_curve(final_rewards, steps=final_steps, success=final_success, window=50, save_path="results/learning_curve.png")
    
    np.save(os.path.join(results_path, "final_q_table.npy"), final_q_table)
    
    score = evaluate_q_table(final_q_table, taxi_env, episodes=100)
    print(f"Reward promedio con política final: {score}")