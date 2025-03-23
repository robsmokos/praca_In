import os
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:\DATA\ROB\PRACA\praca_In\_KOD\\2x2_END_swiatlaAI\\2x2.sumocfg"
MODEL_PATH = "24__BB__.weights.h5"  # Ścieżka do zapisanych wag

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
CONTROL_INTERVAL = 10  # Interwał decyzji

# Definicja modelu
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
        self.common = tf.keras.Sequential([
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(64, activation="relu", kernel_initializer="he_normal")
        ])
        self.actor = layers.Dense(num_tls * num_phases, activation="softmax", name="actor")

    def call(self, state):
        x = self.common(state)
        return self.actor(x)

# Pobieranie stanu
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

# Wybór akcji
def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    return [np.argmax(probs) for probs in action_probs]  # Wybór najlepszej akcji

# Ustawienie świateł
def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# Testowanie modelu
def test_model():
    model = ActorCritic(NUM_TLS, NUM_PHASES)

    # Wczytanie modelu
    if os.path.exists(MODEL_PATH):
        print(f"📥 Wczytywanie modelu z {MODEL_PATH}")
        dummy_input = np.zeros((1, 2 * NUM_TLS), dtype=np.float32)
        model(dummy_input)  # Wymuszenie budowy modelu
        model.load_weights(MODEL_PATH)

    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
    except Exception as e:
        print(f"❌ Błąd uruchamiania SUMO: {e}")
        return

    total_reward = 0.0

    for step in range(7000):
        if step % CONTROL_INTERVAL == 0:
            state = get_state()
            action_probs = model(state)
            actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
            apply_action(actions)

        traci.simulationStep()  # Wykonanie kroku SUMO

    print("✅ Testowanie zakończone.")
    traci.close()

if __name__ == "__main__":
    test_model()
