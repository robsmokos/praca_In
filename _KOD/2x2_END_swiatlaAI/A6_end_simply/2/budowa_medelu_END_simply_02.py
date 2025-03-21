import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Stałe konfiguracyjne
SUMO_BINARY = "sumo"
CONFIG_FILE = "/content/SUMO/2x2_simply.sumocfg"

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
PENALTY = -150
CRITICAL_PENALTY = -50
MAX_CRITICAL_QUEUE = 300
MAX_WAITING_TIME = 10000

# Interwał co ile kroków podejmowana jest NOWA decyzja i aktualizowana sieć
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
    # Normalizacja przykładowa
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

def choose_action(action_probs, num_tls, num_phases, epsilon):
    """Wybór akcji (fazy sygnalizacji) dla każdego skrzyżowania."""
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    # Epsilon-greedy dla większej eksploracji, nawet mimo softmaxa
    if np.random.rand() < epsilon:
        return np.random.randint(0, num_phases, num_tls)
    return [np.random.choice(num_phases, p=probs) for probs in action_probs]

def apply_action(actions):
    """Zastosowanie akcji (stan świateł) w SUMO."""
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

def get_reward():
    """Liczymy nagrodę (lub karę) na podstawie obecnego stanu."""
    total_queue_length = sum(
        sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    total_waiting_time = sum(
        sum(traci.lane.getWaitingTime(lane) for lane in traci.trafficlight.getControlledLanes(tls_id))
        for tls_id in TLS_IDS
    )
    queue_penalty = (total_queue_length / 250) * 0.01
    waiting_penalty = total_waiting_time * 0.00001
    free_flow_bonus = max(0, 1.0 - (queue_penalty + waiting_penalty))
    reward = free_flow_bonus - (queue_penalty * waiting_penalty)

    if total_queue_length > MAX_CRITICAL_QUEUE:
        reward += -(total_queue_length) * 0.001
    if total_waiting_time > MAX_WAITING_TIME:
        reward += -(total_waiting_time - MAX_WAITING_TIME) * 0.00001

    reward -= queue_penalty
    reward -= waiting_penalty
    return np.clip(reward, -1000, 10)

def train_actor_critic():
    model = ActorCritic(NUM_TLS, NUM_PHASES)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    for episode in range(300):
        print(f"\n=== Start epizodu {episode} ===")
        # Bezpieczne zamknięcie poprzedniej sesji TraCI
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

        # Pobieramy stan początkowy (old_state)
        old_state = get_state()
        total_reward = 0.0
        cumulated_reward = 0.0
        epsilon = max(0.01, 0.2 * (0.926 ** episode))

        # --- Na samym początku epizodu: wybieramy i ustawiamy pierwszą akcję ---
        # W modelu: old_state -> action_probs -> old_actions
        action_probs, _ = model(old_state)
        old_actions = choose_action(action_probs, NUM_TLS, NUM_PHASES, epsilon)
        apply_action(old_actions)

        # Symulujemy określoną liczbę kroków
        for step in range(1, 4001):
            # Wykonujemy jeden krok symulacji w SUMO
            traci.simulationStep()

            # Zbieramy reward za ten krok (to efekt *poprzedniej* akcji)
            reward = get_reward()
            cumulated_reward += reward

            # Gromadzimy sumę nagród z ostatniego interwału
            # Jeśli osiągnęliśmy koniec interwału sterowania lub dotarliśmy do końca:
            if (step % CONTROL_INTERVAL == 0) or (step == 4000):
                # Obecny stan - bo będziemy liczyć next_value
                current_state = get_state()

                # Obliczamy target
                _, next_value = model(current_state)
                gamma = 0.95

                # Uwaga: cumulated_reward to nagroda zebrana przez CONTROL_INTERVAL kroków
                # Dla prostoty mnożymy ją tu przez 0.1, ale można to zmodyfikować.
                target = 0.1 * cumulated_reward + gamma * tf.stop_gradient(next_value)

                # Trenujemy na podstawie old_state -> old_actions
                with tf.GradientTape() as tape:
                    # Sieć zwraca action_probs_pred i value_pred dla old_state
                    action_probs_pred, value_pred = model(old_state)

                    # Log-prob wszystkich akcji (dla 4 skrzyżowań x 3 fazy = 12 wyjść)
                    # Tutaj "stary" log_probs odpowiada akcji, jaką podjęliśmy w old_actions
                    log_probs_all = tf.math.log(tf.clip_by_value(action_probs_pred, 1e-8, 1.0))

                    # Aby rozdzielić poszczególne TLS, reshape na [NUM_TLS, NUM_PHASES]
                    log_probs_2d = tf.reshape(log_probs_all, (NUM_TLS, NUM_PHASES))

                    # Musimy wybrać log-prob akcji, którą rzeczywiście wykonaliśmy (old_actions)
                    # Dla każdego TLS mamy jedną fazę z old_actions[i]
                    chosen_log_probs = []
                    for i in range(NUM_TLS):
                        chosen_log_probs.append(log_probs_2d[i, old_actions[i]])
                    chosen_log_probs = tf.stack(chosen_log_probs)

                    # Advantage = target - value_pred
                    advantage = target - value_pred

                    # Actor loss = - mean( log_prob(action) * advantage )
                    actor_loss = -tf.reduce_mean(chosen_log_probs * tf.stop_gradient(advantage))

                    # Critic loss = MSE( target - value_pred )
                    critic_loss = tf.reduce_mean(tf.square(target - value_pred))

                    loss = actor_loss + 0.5 * critic_loss

                # Aktualizacja wag
                grads = tape.gradient(loss, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Dodajemy zsumowaną nagrodę do total_reward (monitoring)
                total_reward += cumulated_reward

                # Zerujemy sumę nagród w interwale
                cumulated_reward = 0.0

                # Przechodzimy do nowego stanu
                old_state = current_state

                if step < 4000:
                    # WYBIERAMY NOWĄ AKCJĘ I USTAWIAMY JĄ (na kolejny interwał)
                    action_probs, _ = model(old_state)
                    old_actions = choose_action(action_probs, NUM_TLS, NUM_PHASES, epsilon)
                    apply_action(old_actions)
                # Jeśli step == 4000 to kończymy epizod

        # Podsumowanie epizodu
        print(f"Epizod {episode} zakończony. Całkowita nagroda: {total_reward:.2f}")

        # Zapis wagi modelu (opcjonalnie z int(total_reward), żeby nie wrzucać kropki itp.)
        model.save_weights(f"/content/drive/MyDrive/SUMO/111_{int(total_reward)}_ep_{episode}__BB__.weights.h5")
        print(f"Model zapisany dla epizodu {episode}")

        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

if __name__ == "__main__":
    train_actor_critic()
