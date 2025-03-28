import os
import csv
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# === SUMO konfiguracja ===
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:\DATA\ROB\PRACA\praca_In\_KOD\\2x2_END_swiatlaAI\\2x2.sumocfg"

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
CONTROL_INTERVAL = 10

# === Definicja modelu ===
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

# === Pobranie stanu (kolejki i czasy) ===
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

# === Wybór akcji (greedy) ===
def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    return [np.random.choice(num_phases, p=probs) for probs in action_probs]

# === Zastosowanie akcji ===
def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# === TESTUJ MODEL I ZAPISZ DO CSV ===
def test_model(model_path, steps=6000, csv_path="test_output.csv"):
    print(f"📦 Ładowanie modelu z: {model_path}")
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model(tf.constant(np.zeros((1, NUM_TLS * 2), dtype=np.float32)))  # Inicjalizacja
    model.load_weights(model_path)

    # Bezpieczne zamknięcie ewentualnych wcześniejszych sesji
    try:
        traci.close()
    except:
        pass

    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])

    # Przygotowanie pliku CSV
    csv_data = []
    csv_data.append(["Krok", "CzasOczekiwania(s)", "ZatrzymanePojazdy", "SredniaPredkosc(m/s)"])

    state = get_state()

    for step in range(steps):
        traci.simulationStep()

        # Zbieranie metryk
        edge_ids = traci.edge.getIDList()
        waiting_times = [traci.edge.getWaitingTime(e) for e in edge_ids if not e.startswith(":")]
        avg_waiting_time = np.mean(waiting_times) if waiting_times else 0.0

        lanes = traci.lane.getIDList()
        num_halted = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in lanes)

        try:
            mean_speed = traci.vehicle.getAverageSpeed() if traci.vehicle.getIDCount() > 0 else 0.0
        except:
            mean_speed = 0.0

        csv_data.append([
            step,
            round(avg_waiting_time, 2),
            num_halted,
            round(mean_speed, 2)
        ])

        # Podejmij akcję co CONTROL_INTERVAL
        if step % CONTROL_INTERVAL == 0 or step == steps - 1:
            action_probs, _ = model(state)
            actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
            apply_action(actions)
            state = get_state()

    traci.close()

    # Zapisz wyniki do CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"✅ Test zakończony. Wyniki zapisano do: {csv_path}")

# === Główne wywołanie ===
if __name__ == "__main__":
    # Zmień na swoją ścieżkę do wag
    model_path = "ep_260_01_BB__.weights.h5"
    test_model(model_path, steps=6000, csv_path="wyniki_testu.csv")
