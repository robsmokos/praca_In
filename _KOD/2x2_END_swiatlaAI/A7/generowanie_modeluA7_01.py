import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import traci

# === KONFIGURACJA ===
SUMO_BINARY = "sumo"
CONFIG_FILE = "/content/SUMO/2x2.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
CONTROL_INTERVAL = 10  # co ile kroków zmieniamy akcję
MODEL_PATH = "/content/drive/MyDrive/SUMO/111_best_ep.weights.h5"
CSV_OUTPUT = "/content/drive/MyDrive/SUMO/test_output.csv"

# === MODEL ===
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

# === FUNKCJE POMOCNICZE ===
def get_state():
    max_queue_length = 400
    max_waiting_time = 10000
    queue_lengths = np.array([
        sum(traci.lane.getLastStepHaltingNumber(lane)
            for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / max_queue_length

    waiting_times = np.array([
        sum(traci.lane.getWaitingTime(lane)
            for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    ], dtype=np.float32) / max_waiting_time

    return np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)

def choose_action(action_probs):
    action_probs = action_probs.numpy().reshape(NUM_TLS, NUM_PHASES)
    action_probs = np.clip(action_probs, 1e-8, 1.0)
    actions = []
    for i in range(NUM_TLS):
        probs = action_probs[i] / np.sum(action_probs[i])
        actions.append(np.random.choice(NUM_PHASES, p=probs))
    return actions

def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# === TESTOWANIE MODEL ===
def test_model():
    print("🧠 Testowanie wytrenowanego modelu...")

    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model.build((None, NUM_TLS * 2))
    model.load_weights(MODEL_PATH)
    print(f"✅ Wczytano model z: {MODEL_PATH}")

    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
    except Exception as e:
        print(f"❌ Błąd uruchamiania SUMO: {e}")
        return

    # === CSV OUTPUT ===
    with open(CSV_OUTPUT, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Krok", "CzasOczekiwania(s)", "ZatrzymanePojazdy", "SredniaPredkosc(m/s)"])

        state = get_state()
        actions = choose_action(model(state)[0])
        apply_action(actions)

        for step in range(6000):
            traci.simulationStep()

            # Zbieranie danych:
            total_waiting_time = sum(
                sum(traci.lane.getWaitingTime(lane)
                    for lane in traci.trafficlight.getControlledLanes(tls_id))
                for tls_id in TLS_IDS
            )

            total_halted = sum(
                sum(traci.lane.getLastStepHaltingNumber(lane)
                    for lane in traci.trafficlight.getControlledLanes(tls_id))
                for tls_id in TLS_IDS
            )

            all_speeds = [traci.edge.getLastStepMeanSpeed(edge)
                          for edge in traci.edge.getIDList()
                          if not edge.startswith(":")]
            average_speed = np.mean(all_speeds) if all_speeds else 0.0

            # Zapis do CSV
            writer.writerow([
                step,
                round(total_waiting_time, 2),
                total_halted,
                round(average_speed, 2)
            ])

            # Co kontrol interval → nowa akcja
            if step % CONTROL_INTERVAL == 0:
                state = get_state()
                actions = choose_action(model(state)[0])
                apply_action(actions)

    traci.close()
    print(f"✅ Wyniki zapisane do: {CSV_OUTPUT}")

if __name__ == "__main__":
    test_model()
