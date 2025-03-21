import os
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:/DATA/ROB/PRACA/praca_In/_KOD/2x2_END_swiatlaAI/A6_end_simply/2x2_simply.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 4
MAX_CRITICAL_QUEUE = 300  # Krytyczny poziom korka
MAX_WAITING_TIME = 10000  # Maksymalny czas oczekiwania

# 🔹 Model Actor-Critic
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
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

# 🔹 Pobieranie stanu skrzyżowań
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

    return np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)

# 🔹 Wybór akcji
def choose_action(action_probs):
    action_probs = action_probs.numpy().reshape(NUM_TLS, NUM_PHASES)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(NUM_TLS):
        row_sum = np.sum(action_probs[i])
        action_probs[i] = action_probs[i] / row_sum if row_sum > 0 else np.ones(NUM_PHASES) / NUM_PHASES

    return [np.random.choice(NUM_PHASES, p=probs) for probs in action_probs]

# 🔹 Ustawianie faz świateł
def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# 🔹 Obliczanie nagrody
def get_reward():
    total_queue_length = sum(
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )

    total_waiting_time = sum(
        sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )

    queue_penalty = (total_queue_length / 250) * 0.01
    waiting_penalty = (total_waiting_time) * 0.00001
    free_flow_bonus = max(0, 1.0 - (queue_penalty + waiting_penalty))
    reward = free_flow_bonus - (queue_penalty * waiting_penalty)

    # Kara za duży korek
    if total_queue_length > MAX_CRITICAL_QUEUE:
        reward -= (total_queue_length * 0.001)

    # Kara za długi czas oczekiwania
    if total_waiting_time > MAX_WAITING_TIME:
        reward -= (total_waiting_time - MAX_WAITING_TIME) * 0.00001

    reward = np.clip(reward, -1000, 10)
    return reward

# 🔹 Testowanie modelu
def test_model(model_path, steps=7000):
    print(f"🔍 Testowanie modelu z wagami: {model_path}")

    model = ActorCritic(NUM_TLS, NUM_PHASES)

    # 🔹 Poprawione ładowanie wag modelu
    dummy_input = np.zeros((1, NUM_TLS * 2), dtype=np.float32)
    _ = model(dummy_input)  # Forward pass do inicjalizacji modelu
    model.load_weights(model_path)
    print("✅ Wagi modelu załadowane!")

    # 🔹 Uruchamianie SUMO
    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
    except Exception as e:
        print(f"❌ Błąd uruchamiania SUMO: {e}")
        return

    total_reward = 0

    for step in range(steps):
        state = get_state()
        action_probs, _ = model(state)
   

        if step % 10 == 0:
            actions = choose_action(action_probs)
            apply_action(actions)


        traci.simulationStep()

        reward = get_reward()
        total_reward += reward

        print(f"🔄 Step {step}: Wybrane akcje -> {actions}, Nagroda: {reward}")

        # 🔹 Wcześniejsze zakończenie testu, jeśli ruch jest płynny
        if reward == 1.0 and step > 100:
            print(f"✅ Płynny ruch wykryty w kroku {step} (queue=0, wait=0.0). Reset testu!")
            break

    print(f"🎯 Całkowita nagroda w teście: {total_reward}")
    traci.close()

# 🔹 Uruchomienie testowania
if __name__ == "__main__":
    test_model("epizod_69_A_.weights.h5")
    
    
## 36   7000>
## 47   7000>
## 62   7000>