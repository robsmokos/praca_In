import os
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:/DATA/ROB/PRACA/praca_In/_KOD/2x2_END_swiatlaAI/2x2.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 4

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
def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    actions = [np.random.choice(num_phases, p=probs) for probs in action_probs]
    return actions

# 🔹 Ustawianie faz świateł
def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# 🔹 Testowanie modelu
def test_model(model_path, steps=7000):
    print(f"🔍 Testowanie modelu z wagami: {model_path}")

    model = ActorCritic(NUM_TLS, NUM_PHASES)

    # 🔹 WAŻNE: Najpierw wykonujemy forward pass na sztucznym wejściu, aby poprawnie zbudować model
    dummy_input = np.zeros((1, NUM_TLS * 2), dtype=np.float32)  # Dopasowujemy wejście do kształtu modelu
    _ = model(dummy_input)  # Wykonujemy wywołanie, aby zainicjalizować warstwy

    model.load_weights(model_path)
    print("✅ Wagi modelu załadowane!")

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
            actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
            apply_action(actions)            
        
        
        traci.simulationStep()
        print(f"🔄 Step {step}: Wybrane akcje -> {actions}")

    print("✅ Test zakończony!")
    traci.close()

# 🔹 Uruchomienie testowania
if __name__ == "__main__":
    test_model("epizod_42.weights.h5")


##25   7000>
##36   7000>
##42   6100
##52   5400
##60   6100
