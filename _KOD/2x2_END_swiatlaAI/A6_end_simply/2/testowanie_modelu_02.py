import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:\DATA\ROB\PRACA\praca_In\_KOD\\2x2_END_swiatlaAI\A6_end_simply\\2x2_simply.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3

# Parametry normalizacji
MAX_QUEUE_LENGTH = 400
MAX_WAITING_TIME = 10000
CONTROL_INTERVAL = 10

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
        return self.actor(x), self.critic(x)

def get_state():
    """Pobranie aktualnego stanu środowiska"""
    queue_lengths = np.array([
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / MAX_QUEUE_LENGTH

    waiting_times = np.array([
        sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / MAX_WAITING_TIME

    return np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)

def choose_action(action_probs, num_tls, num_phases):
    """Wybór akcji na podstawie nauczonych polityk"""
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    return [np.random.choice(num_phases, p=probs) for probs in action_probs]

def apply_action(actions):
    """Zastosowanie akcji w SUMO"""
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

def test_model(weights_path, test_episodes=10):
    """Testowanie modelu w SUMO"""
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model.build(input_shape=(None, NUM_TLS * 2))
    model.load_weights(weights_path)
    print(f"✅ Model załadowany: {weights_path}")

    for episode in range(test_episodes):
        print(f"\n=== Start testu epizodu {episode} ===")

        # Zamknięcie ewentualnej poprzedniej sesji TraCI
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        # Uruchamiamy SUMO
        try:
            traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
        except Exception as e:
            print(f"❌ Błąd uruchamiania SUMO: {e}")
            continue

        state = get_state()
        total_reward = 0.0

        for step in range(1, 4001):
            traci.simulationStep()

            # Pobranie decyzji modelu
            action_probs, _ = model(state)
            actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
            apply_action(actions)

            # Obserwacja nowego stanu
            state = get_state()

            # Monitoring
            if step % 500 == 0:
                print(f"Step {step}: Test w toku...")

        print(f"Epizod {episode} zakończony.")

        # Zamknięcie SUMO po zakończeniu epizodu
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

if __name__ == "__main__":
    test_model("ep_19__BB__.weights.h5")
