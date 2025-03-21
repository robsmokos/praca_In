 #c:\DATA\ROB\PRACA\praca_In\_KOD\2x2_END_swiatlaAI\A6_end_simply\1

import os
import sys
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Stałe konfiguracyjne
SUMO_BINARY = "sumo"
#CONFIG_FILE = "/content/SUMO/2x2.sumocfg"
CONFIG_FILE = "c:\DATA\ROB\PRACA\praca_In\_KOD\2x2_END_swiatlaAI\A6_end_simply\2x2.sumocfg"

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 4
UNCHANGE_LIMIT = 50
FORCED_DURATION = 30
PENALTY = -150
CRITICAL_PENALTY = -50
MAX_CRITICAL_QUEUE = 300
MAX_WAITING_TIME = 10000

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
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    if np.random.rand() < epsilon:
        return np.random.randint(0, num_phases, num_tls)
    return [np.random.choice(num_phases, p=probs) for probs in action_probs]

def apply_action(actions, step):
    phases = ["GGgrrrGGgrrr", "yyyyyyyyyyyy", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

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
    waiting_penalty = total_waiting_time * 0.00001
    free_flow_bonus = max(0, 1.0 - (queue_penalty + waiting_penalty))
    reward = free_flow_bonus - (queue_penalty * waiting_penalty)

    if forced_steps > 0:
        reward += PENALTY
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
        print(f"\n🔔 Start epizodu {episode}")
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
        epsilon = max(0.01, 0.2 * (0.926 ** episode))

        for step in range(4000):
            if step % 10 == 0:
                if forced_steps > 0:
                    actions = np.random.randint(0, NUM_PHASES, NUM_TLS)
                    forced_steps -= 10
                else:
                    action_probs, _ = model(state)
                    actions = choose_action(action_probs, NUM_TLS, NUM_PHASES, epsilon)
                apply_action(actions, step)

            traci.simulationStep()
            next_state = get_state()

            if step % 10 == 0:
                reward = get_reward(forced_steps, step)
                total_reward += reward

                _, next_value = model(next_state)
                gamma = 0.95
                target = reward + gamma * tf.stop_gradient(next_value)

                if 1 <= step < 6997:
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
        model.save_weights(f"/content/drive/MyDrive/SUMO/300_{total_reward}model_epizod_{episode}__BB__.weights.h5")
        print(f"💾 Model zapisany dla epizodu {episode}")

        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

if __name__ == "__main__":
    train_actor_critic()