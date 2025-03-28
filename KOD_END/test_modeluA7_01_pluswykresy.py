import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import traci

# === Stałe ===
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:\DATA\ROB\PRACA\praca_In\_KOD\\2x2_END_swiatlaAI\\2x2.sumocfg"
MODEL_WEIGHTS = "200__BB_01_.weights.h5"
CSV_OUTPUT_PATH = "test_modelu_AI_01.csv"

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
CONTROL_INTERVAL = 10
MAX_STEPS = 6000

# === Model ===
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

# === Pomocnicze ===
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

def choose_action(action_probs):
    action_probs = action_probs.numpy().reshape(NUM_TLS, NUM_PHASES)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(NUM_TLS):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(NUM_PHASES) / NUM_PHASES
        else:
            action_probs[i] /= row_sum

    return [np.random.choice(NUM_PHASES, p=probs) for probs in action_probs]

def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

def get_step_metrics():
    waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in traci.lane.getIDList())
    stopped_vehicles = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.lane.getIDList())
    speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in traci.vehicle.getIDList()]
    avg_speed = np.mean(speeds) if speeds else 0.0
    return waiting_time, stopped_vehicles, avg_speed

# === Główna funkcja testująca ===
def test_model():
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model(tf.convert_to_tensor(np.zeros((1, NUM_TLS * 2), dtype=np.float32)))  # inicjalizacja
    model.load_weights(MODEL_WEIGHTS)

    # Zamknij poprzednie połączenie TraCI jeśli trzeba
    try:
        traci.close()
    except Exception:
        pass

    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])

    with open(CSV_OUTPUT_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Krok", "CzasOczekiwania(s)", "ZatrzymanePojazdy", "SredniaPredkosc(m/s)"])

        state = get_state()
        action_probs, _ = model(state)
        actions = choose_action(action_probs)
        apply_action(actions)

        for step in range(MAX_STEPS):
            traci.simulationStep()

            # Zbieramy dane
            waiting_time, stopped_vehicles, avg_speed = get_step_metrics()
            writer.writerow([step, round(waiting_time, 2), stopped_vehicles, round(avg_speed, 2)])

            # Co CONTROL_INTERVAL wybieramy nową akcję
            if (step + 1) % CONTROL_INTERVAL == 0:
                state = get_state()
                action_probs, _ = model(state)
                actions = choose_action(action_probs)
                apply_action(actions)

    traci.close()
    print(f"✅ Test zakończony. Dane zapisane w: {CSV_OUTPUT_PATH}")

# === Uruchomienie ===
if __name__ == "__main__":
    test_model()
