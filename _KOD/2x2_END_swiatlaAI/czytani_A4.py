import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo-gui"            # lub "sumo-gui", jeśli chcesz oglądać symulację
CONFIG_FILE = "2x2.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]  # Lista sygnalizatorów
NUM_TLS = len(TLS_IDS)          # Liczba sygnalizatorów
NUM_PHASES = 4                  # Liczba faz dla każdego sygnalizatora

UNCHANGE_LIMIT = 50             # Limit kroków bez zmiany fazy
FORCED_DURATION = 30            # Czas wymuszonej losowej fazy
PENALTY = -0.1                  # Kara za wymuszenie losowej fazy

# 🔹 Model Actor-Critic (identyczny jak w treningu)
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
        self.num_tls = num_tls
        self.num_phases = num_phases
        self.common = tf.keras.Sequential([
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(64, activation="relu",  kernel_initializer="he_normal")
        ])
        self.actor = layers.Dense(num_tls * num_phases, activation="softmax", name="actor")
        self.critic = layers.Dense(1, name="critic")

    def call(self, state):
        x = self.common(state)
        actor_output = self.actor(x)
        critic_output = self.critic(x)
        return actor_output, critic_output

def get_state():
    # Zgodnie z Twoją definicją w treningu
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

    state = np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)
    return state

def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] = action_probs[i] / row_sum

    actions = [np.random.choice(num_phases, p=probs) for probs in action_probs]
    return actions

def apply_action(actions, step):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])
    print(f"🟢 Step {step}: Ustawiono fazy {actions}")

def get_reward(forced_steps):
    total_queue_length = sum(
        sum(traci.lane.getLastStepHaltingNumber(lane) 
            for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    total_waiting_time = sum(
        sum(traci.lane.getWaitingTime(lane) 
            for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    queue_penalty = total_queue_length / 250  
    waiting_penalty = total_waiting_time / 1000  
    free_flow_bonus = 1.0 - (queue_penalty + waiting_penalty)
    reward = free_flow_bonus - (queue_penalty + 0.5 * waiting_penalty)
    if forced_steps > 0:
        reward += PENALTY
    return reward

def test_model(weights_path, simulation_steps=5000):
    """
    Uruchamia symulację SUMO z wczytanym modelem
    i pozwala agentowi sterować fazami świateł bez żadnego treningu.
    """
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    model(tf.zeros((1, 8)))  # "dummy call"
    model.load_weights(weights_path)


    # 1. Inicjujemy model i wczytujemy wagi
    #model = ActorCritic(NUM_TLS, NUM_PHASES)
    #model.load_weights(weights_path)
    print(f"✅ Załadowano wagi z pliku: {weights_path}")

    # 2. Uruchamiamy SUMO
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE])

    try:
        state = get_state()
        total_reward = 0.0
        forced_steps = 0
        actions = np.zeros(NUM_TLS, dtype=int)  # Początkowa faza (domyślnie 0)
        unchanged_steps_global = 0

        # 3. Główna pętla symulacji
        for step in range(simulation_steps):
            # Co 10 kroków ewentualnie zmieniamy fazy świateł:
            if step % 10 == 0:
                if forced_steps > 0:
                    new_actions = np.random.randint(0, NUM_PHASES, NUM_TLS)
                    forced_steps -= 10
                else:
                    action_probs, _ = model(state)
                    new_actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
                    if np.array_equal(new_actions, actions):
                        unchanged_steps_global += 1
                    else:
                        unchanged_steps_global = 0

                    if unchanged_steps_global >= UNCHANGE_LIMIT:
                        new_actions = np.random.randint(0, NUM_PHASES, NUM_TLS)
                        forced_steps = FORCED_DURATION
                        unchanged_steps_global = 0
                        print(f"🚨 Wymuszona losowa faza {new_actions} przez {FORCED_DURATION} kroków! Kara {PENALTY}")

                apply_action(new_actions, step)
                actions = new_actions

            # Jeden krok SUMO
            traci.simulationStep()

            # Obliczamy stan, nagrodę (tylko do podglądu)
            next_state = get_state()
            reward = get_reward(forced_steps)
            total_reward += reward

            # Stan = next_state (bez uczenia)
            state = next_state
        
        print(f"✅ Zakończono testowanie. Łączna nagroda: {total_reward:.2f}")

    finally:
        traci.close()

# 🔹 Uruchomienie testu z konkretnym plikiem wag
if __name__ == "__main__":
    # Przykładowa ścieżka do pliku wag (dostosuj do swojej nazwy/ścieżki):
    weights_file = "model_epizod_15.weights.h5"
    
    # Liczba kroków w teście (np. 5000)
    test_steps = 5000

    test_model(weights_path=weights_file, simulation_steps=test_steps)
