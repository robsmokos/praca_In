import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo"            # Możesz użyć "sumo-gui", jeśli chcesz oglądać symulację
CONFIG_FILE = "/content/SUMO/2x2.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]  # Lista sygnalizatorów
NUM_TLS = len(TLS_IDS)          # Liczba sygnalizatorów
NUM_PHASES = 4                  # Liczba faz dla każdego sygnalizatora
UNCHANGE_LIMIT = 50             # Limit kroków bez zmiany fazy
FORCED_DURATION = 30            # Czas wymuszonej losowej fazy
PENALTY = -0.1                  # Kara za wymuszenie losowej fazy

# 🔹 Model Actor-Critic
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
        self.num_tls = num_tls
        self.num_phases = num_phases
        # Zmieniamy warstwę common na trzy Dense:
        self.common = tf.keras.Sequential([
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(64, activation="relu",  kernel_initializer="he_normal")
        ])
        # Warstwa aktora: wyjście o rozmiarze (num_tls * num_phases)
        self.actor = layers.Dense(num_tls * num_phases, activation="softmax", name="actor")
        # Warstwa krytyka: pojedyncza wartość
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

# 🔹 Funkcja wyboru akcji (poprawiona wersja)
def choose_action(action_probs, num_tls, num_phases):
    # Konwersja tensora na tablicę NumPy oraz przekształcenie do kształtu (num_tls, num_phases)
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    
    # Klipowanie wartości ujemnych – jeśli któreś prawdopodobieństwo jest ujemne, ustawiamy je na 0
    action_probs = np.clip(action_probs, 0, None)
    
    # Normalizacja każdego wiersza, aby suma wynosiła dokładnie 1
    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] = action_probs[i] / row_sum

    # Wybór akcji (fazy) dla każdego sygnalizatora
    actions = [np.random.choice(num_phases, p=probs) for probs in action_probs]
    return actions

# 🔹 Funkcja ustawiająca fazy świateł
def apply_action(actions, step):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])
    print(f"🟢 Step {step}: Ustawiono fazy {actions}")

# 🔹 Funkcja obliczająca nagrodę
def get_reward(forced_steps):
    total_queue_length = sum(
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    total_waiting_time = sum(
        sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    queue_penalty = total_queue_length / 250  
    waiting_penalty = total_waiting_time / 1000  
    free_flow_bonus = 1.0 - (queue_penalty + waiting_penalty)
    reward = free_flow_bonus - (queue_penalty + 0.5 * waiting_penalty)
    if forced_steps > 0:
        reward += PENALTY
    return reward

# 🔹 Funkcja treningowa z mechanizmem losowego uruchamiania uczenia przez 1000 kroków
def train_actor_critic():
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    simulation_steps = 5000      # Całkowita liczba kroków symulacji w epizodzie
    learning_duration = 1000     # Model będzie aktualizowany przez dokładnie 1000 kroków SUMO

    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE])
        for episode in range(30):
            state = get_state()
            total_reward = 0
            forced_steps = 0                    # Licznik kroków, gdy wymuszana jest losowa faza
            actions = np.zeros(NUM_TLS, dtype=int)  # Aktualnie ustawione fazy
            unchanged_steps_global = 0          # Licznik kroków bez zmiany fazy

            # Losujemy liczbę R z przedziału 1-4000, od której rozpocznie się aktualizacja wag
            start_training_step = np.random.randint(1, 4001)
            end_training_step = start_training_step + learning_duration
            print(f"🔔 Epizod {episode}: uczenie bedzie aktywne od kroku {start_training_step} do {end_training_step - 1}")

            for step in range(simulation_steps):
                # Co 10 kroków następuje ewentualna zmiana faz świateł
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
                            print(f"🚨 Wymuszona losowa faza {new_actions} przez {FORCED_DURATION} kroków! Dodana kara {PENALTY}")

                    apply_action(new_actions, step)
                    actions = new_actions

                traci.simulationStep()
                next_state = get_state()
                reward = get_reward(forced_steps)
                total_reward += reward

                # Obliczamy wartość stanu następnego
                _, next_value = model(next_state)
                gamma = 0.98
                target = reward + gamma * tf.stop_gradient(next_value)
                
                # Aktualizujemy model tylko, jeśli bieżący krok mieści się w przedziale [start_training_step, end_training_step)
                if start_training_step <= step < end_training_step:
                    with tf.GradientTape() as tape:
                        action_probs_pred, value_pred = model(state)
                        log_probs = tf.math.log(tf.clip_by_value(action_probs_pred, 1e-8, 1.0))
                        
                        # Przygotowanie tensorów dla wybranych akcji
                        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.int32)
                        actions_tensor = tf.expand_dims(actions_tensor, axis=0)
                        actions_one_hot = tf.one_hot(actions_tensor, depth=NUM_PHASES, dtype=tf.float32)
                        actions_one_hot = tf.reshape(actions_one_hot, [1, NUM_TLS * NUM_PHASES])
                        
                        selected_log_probs = tf.reduce_sum(log_probs * actions_one_hot, axis=1)
                        advantage = target - value_pred
                        
                        actor_loss = -tf.reduce_mean(selected_log_probs * tf.stop_gradient(advantage))
                        critic_loss = tf.reduce_mean(tf.square(target - value_pred))
                        loss = actor_loss + 0.5 * critic_loss
                    
                    grads = tape.gradient(loss, model.trainable_variables)
                    grads, _ = tf.clip_by_global_norm(grads, 5.0)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                # Aktualizujemy stan na kolejny krok
                state = next_state

            print(f"✅ Epizod {episode}, Całkowita nagroda: {total_reward}")
            model.save_weights(f"/content/SUMO/model3/{total_reward}model_epizod_{episode}.weights.h5")
            print(f"💾 Model zapisany dla epizodu {episode}")

    finally:
        traci.close()

# 🔹 Uruchamiamy trening modelu
if __name__ == "__main__":
    train_actor_critic()
