# simpy
import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# 🔹 Stałe konfiguracyjne
SUMO_BINARY = "sumo"
CONFIG_FILE = "/content/SUMO/2x2_simply.sumocfg"
TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 4
UNCHANGE_LIMIT = 50
FORCED_DURATION = 30
PENALTY = -150  # Zwiększona kara za wymuszone fazy!
CRITICAL_PENALTY = -50  # Duża kara za totalny korek
MAX_CRITICAL_QUEUE = 300  # Definiujemy krytyczny poziom korka
MAX_WAITING_TIME = 10000  # Maksymalny czas oczekiwania przed nałożeniem kary

# 🔹 Model Actor-Critic
class ActorCritic(tf.keras.Model):
    def __init__(self, num_tls, num_phases):
        super().__init__()
        self.num_tls = num_tls
        self.num_phases = num_phases
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

    # Debugowanie wartości kolejek i czasu oczekiwania (zakomentowane)
    #print(f"📊 Debug - queue_lengths: {queue_lengths * max_queue_length}")
    #print(f"📊 Debug - waiting_times: {waiting_times * max_waiting_time}")
    #print(f"✅ Debug - queue_lengths (normalized): {queue_lengths}")
    #print(f"✅ Debug - waiting_times (normalized): {waiting_times}")

    return np.concatenate([queue_lengths, waiting_times]).reshape(1, -1)

# 🔹 Wybór akcji (epsilon-greedy) z przekazywanym epsilon
def choose_action(action_probs, num_tls, num_phases, epsilon):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    # Mechanizm epsilon-greedy: z prawdopodobieństwem epsilon wybierana jest losowa akcja
    if np.random.rand() < epsilon:
        actions = np.random.randint(0, num_phases, num_tls)
    else:
        actions = [np.random.choice(num_phases, p=probs) for probs in action_probs]

    return actions

# 🔹 Funkcja ustawiająca fazy świateł
def apply_action(actions, step):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# 🔹 Obliczanie nagrody
def get_reward(forced_steps, step):
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

    # Kara za wymuszone fazy świateł
    if forced_steps > 0:
        reward += PENALTY

    # Dynamiczna kara za duży korek
    if total_queue_length > MAX_CRITICAL_QUEUE:
        dynamic_penalty = -(total_queue_length) * 0.001  # Kara proporcjonalna do nadmiaru
        reward += dynamic_penalty

    # Kara za długi czas oczekiwania (dynamiczna)
    if total_waiting_time > MAX_WAITING_TIME:
        extra_penalty = -(total_waiting_time - MAX_WAITING_TIME) * 0.00001  # Proporcjonalna kara
        reward += extra_penalty

    # Kara za długość kolejki (zawsze działa)
    reward -= queue_penalty

    # Kara za długi czas oczekiwania (zawsze działa)
    reward -= waiting_penalty

    # Ograniczenie nagród
    reward = np.clip(reward, -1000, 10)

    return reward



def train_actor_critic():
    model = ActorCritic(NUM_TLS, NUM_PHASES)

    # 🔹 Ścieżka do zapisanego modelu
    model_path = "/content/drive/MyDrive/SUMO/82.31272207760045model_epizod_84.weights.h5"

    # 🔹 Sprawdzenie, czy istnieją zapisane wagi, i ich wczytanie
    if os.path.exists(model_path):
        print(f"📥 Wczytywanie istniejących wag z {model_path}")

    # 🔹 Tworzymy sztuczne wejście do wymuszenia budowy modelu
        dummy_input = np.zeros((1, 2 * NUM_TLS), dtype=np.float32)  # Dopasowane do get_state()
    
    # 🔹 Wymuszamy budowę modelu
        model(dummy_input) 
        model.load_weights(model_path)  # Teraz można wczytać wagi
        

    # Wybór optymalizatora (zmniejszona wartość learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    for episode in range(100):
        print(f"\n🔔 Start epizodu {episode}")

        # Restart SUMO przed każdym epizodem
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        try:
            traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
        except Exception as e:
            print(f"❌ Błąd uruchamiania SUMO: {e}")
            continue

        state = get_state()
        total_reward = 0
        forced_steps = 0

        # Dynamiczna aktualizacja wartości epsilon
        # epsilon = max(0.01, 0.2 * (0.926 ** episode))  # 40 epizodów do osiągnięcia 0.01
        epsilon = 0.01


        start_training_step = np.random.randint(1, 3)
        end_training_step = start_training_step + 3297
        print(f"🔄 Epizod {episode}: Uczenie od kroku {start_training_step} do {end_training_step - 1}")

        for step in range(3300):
            if step % 10 == 0:
                if forced_steps > 0:
                    actions = np.random.randint(0, NUM_PHASES, NUM_TLS)
                    forced_steps -= 10
                else:
                    action_probs, _ = model(state)
                    actions = choose_action(action_probs, NUM_TLS, NUM_PHASES, epsilon)

                apply_action(actions, step)
                # print(f"🔄 Step {step}: Wybrane akcje -> {actions}")

            traci.simulationStep()
            next_state = get_state()
            reward = get_reward(forced_steps, step)

            # Reset epizodu po 10 krokach płynnego ruchu
            if reward == 1.0 and step > 100  and step < 2000 :
                print(f"✅ Płynny ruch wykryty w kroku {step} (queue=0, wait=0.0). Reset epizodu!")
                break

            total_reward += reward

            _, next_value = model(next_state)
            gamma = 0.95
            target = reward + gamma * tf.stop_gradient(next_value)

            if start_training_step <= step < end_training_step:
                with tf.GradientTape() as tape:
                    action_probs_pred, value_pred = model(state)
                    log_probs = tf.math.log(tf.clip_by_value(action_probs_pred, 1e-8, 1.0))
                    advantage = target - value_pred
                    actor_loss = -tf.reduce_mean(log_probs * tf.stop_gradient(advantage))
                    critic_loss = tf.reduce_mean(tf.square(target - value_pred))
                    loss = actor_loss + 0.5 * critic_loss

                grads = tape.gradient(loss, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            state = next_state

        print(f"✅ Epizod {episode} zakończony, Całkowita nagroda: {total_reward}")
        model.save_weights(f"/content/drive/MyDrive/SUMO/{total_reward}model_epizod_{episode}_A_.weights.h5")
        print(f"💾 Model zapisany dla epizodu {episode}")

        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

# 🔹 Uruchomienie treningu
if __name__ == "__main__":
    train_actor_critic()
