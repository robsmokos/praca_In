import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"  # Możesz użyć "sumo", jeśli nie potrzebujesz interfejsu graficznego
CONFIG_FILE = "2x2.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]  # Lista sygnalizatorów
NUM_TLS = len(TLS_IDS)  # Liczba sygnalizatorów
NUM_PHASES = 4  # Liczba faz dla każdego sygnalizatora

# 🔹 Model Actor-Critic
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.common = tf.keras.Sequential([
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal")
        ])
        self.actor = layers.Dense(num_tls * num_phases, activation="softmax", name="actor")
        self.critic = layers.Dense(1, name="critic")

    def call(self, state):
        x = self.common(state)
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

# 🔹 Funkcja pobierająca stan skrzyżowań
def get_state():
    max_queue_length = 400
    max_waiting_time = 10000

    queue_lengths = np.array([
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / max_queue_length

    waiting_times = np.array([
        sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / max_waiting_time

    state = np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)
    return state

# 🔹 Funkcja wyboru akcji
def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    
    # Normalizacja prawdopodobieństw
    action_probs = np.clip(action_probs, 1e-8, 1.0)  # Ogranicz wartości do zakresu [1e-8, 1.0]
    action_probs = action_probs / np.sum(action_probs, axis=1, keepdims=True)
    
    # Sprawdź, czy prawdopodobieństwa zawierają NaN
    if np.any(np.isnan(action_probs)):
        print("⚠️ Wykryto NaN w prawdopodobieństwach! Ustawiam domyślne prawdopodobieństwa.")
        action_probs = np.ones_like(action_probs) / num_phases  # Ustaw równomierne prawdopodobieństwa

    # Wybierz fazę dla każdego sygnalizatora
    actions = [np.random.choice(num_phases, p=probs) for probs in action_probs]
    return actions

# 🔹 Funkcja ustawiająca fazy świateł
def apply_action(actions, step):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])
    print(f"Step {step}: actions = {actions}")

# 🔹 Funkcja testująca model
def test_model(model_path, num_episodes=5, simulation_steps=5000):
    # Wczytanie modelu
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model.build(input_shape=(None, 2 * NUM_TLS))  # Zbuduj model z odpowiednim kształtem wejścia
    model.load_weights(model_path)  # Wczytaj wagi modelu
    print(f"💾 Wczytano wagi modelu z pliku: {model_path}")

    # Uruchomienie SUMO
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "-e", str(simulation_steps)])
    try:
        for episode in range(num_episodes):
            state = get_state()

            for step in range(simulation_steps):
                if step % 10 == 0:  # Zmiana fazy co 10 kroków
                    action_probs, _ = model(state)  # Użyj modelu do wygenerowania prawdopodobieństw akcji
                    actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)  # Wybierz akcje
                    apply_action(actions, step)  # Ustaw fazy świateł

                traci.simulationStep()  # Wykonaj krok symulacji
                state = get_state()  # Pobierz nowy stan

            print(f"✅ Epizod {episode} zakończony.")
    finally:
        traci.close()

# 🔹 Uruchomienie testowania modelu
if __name__ == "__main__":
    MODEL_PATH = "model_epizod_29.weights.h5"  # Podaj ścieżkę do pliku z wagami modelu
    test_model(MODEL_PATH, num_episodes=5, simulation_steps=5000)
