import os
import traci
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# === Konfiguracja ===
SUMO_BINARY = "sumo-gui"
CONFIG_FILE = "c:\\DATA\\ROB\\PRACA\\praca_In\\_KOD\\2x2_END_swiatlaAI\\2x2.sumocfg"
MODEL_PATH = "54__BB__.weights.h5"

TLS_IDS = ["P4", "P5", "P8", "P9"]
NUM_TLS = len(TLS_IDS)
NUM_PHASES = 3
CONTROL_INTERVAL = 10

# === Model ===
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

# === Pobieranie stanu ===
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

# === Wybór akcji ===
def choose_action(action_probs, num_tls, num_phases):
    action_probs = action_probs.numpy().reshape(num_tls, num_phases)
    action_probs = np.clip(action_probs, 0, None)

    for i in range(num_tls):
        row_sum = np.sum(action_probs[i])
        if np.isclose(row_sum, 0.0):
            action_probs[i] = np.ones(num_phases) / num_phases
        else:
            action_probs[i] /= row_sum

    return [np.argmax(probs) for probs in action_probs]

# === Ustawienie świateł ===
def apply_action(actions):
    phases = ["GGgrrrGGgrrr", "rrrGGgrrrGGg", "GGGGGGGGGGGG"]
    for tls_id, action in zip(TLS_IDS, actions):
        traci.trafficlight.setRedYellowGreenState(tls_id, phases[action])

# === Testowanie modelu ===
def test_model():
    model = ActorCritic(NUM_TLS, NUM_PHASES)

    if os.path.exists(MODEL_PATH):
        print(f"📥 Wczytywanie modelu z {MODEL_PATH}")
        dummy_input = np.zeros((1, 2 * NUM_TLS), dtype=np.float32)
        model(dummy_input)
        model.load_weights(MODEL_PATH)

    try:
        traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start"])
    except Exception as e:
        print(f"❌ Błąd uruchamiania SUMO: {e}")
        return

    avg_waiting_times = []
    total_halted_vehicles = []
    vehicle_speeds = []

    total_waiting_time_all_vehicles = 0.0
    total_vehicles_seen = set()

    for step in range(5000):
        if step % CONTROL_INTERVAL == 0:
            state = get_state()
            action_probs = model(state)
            actions = choose_action(action_probs, NUM_TLS, NUM_PHASES)
            apply_action(actions)

        traci.simulationStep()

        # Zbieranie metryk
        waiting_time = 0.0
        halted = 0
        for tls_id in TLS_IDS:
            lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane in lanes:
                try:
                    waiting_time += traci.lane.getWaitingTime(lane)
                    halted += traci.lane.getLastStepHaltingNumber(lane)
                except traci.exceptions.TraCIException:
                    pass

        avg_waiting_times.append(waiting_time)
        total_halted_vehicles.append(halted)

        # Prędkość pojazdów
        vehicle_ids = traci.vehicle.getIDList()
        speeds = [traci.vehicle.getSpeed(veh_id) for veh_id in vehicle_ids]
        vehicle_speeds.append(np.mean(speeds) if speeds else 0.0)

        # Zbieranie czasu oczekiwania pojazdów
        for veh_id in vehicle_ids:
            try:
                total_waiting_time_all_vehicles += traci.vehicle.getWaitingTime(veh_id)
                total_vehicles_seen.add(veh_id)
            except traci.exceptions.TraCIException:
                pass

    traci.close()
    print("✅ Testowanie zakończone.")

    # Statystyki końcowe
    mean_waiting = np.mean(avg_waiting_times)
    mean_speed = np.mean(vehicle_speeds)
    num_vehicles = len(total_vehicles_seen)
    avg_waiting_per_vehicle = total_waiting_time_all_vehicles / num_vehicles if num_vehicles > 0 else 0.0

    print("\n📊 Statystyki końcowe:")
    print(f"🔵 Średni czas oczekiwania (sumaryczny): {mean_waiting:.2f} s")
    print(f"🟢 Średnia prędkość pojazdów: {mean_speed:.2f} m/s")
    print(f"🚗 Średni czas oczekiwania pojedynczego pojazdu: {avg_waiting_per_vehicle:.2f} s")

    # === Zapis do pliku txt ===
    output_path = "wyniki_symulacji.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Krok\tCzasOczekiwania(s)\tZatrzymanePojazdy\tSredniaPredkosc(m/s)\n")
        for i in range(len(avg_waiting_times)):
            f.write(f"{i}\t{avg_waiting_times[i]:.2f}\t{total_halted_vehicles[i]}\t{vehicle_speeds[i]:.2f}\n")

        f.write("\n--- Statystyki końcowe ---\n")
        f.write(f"Średni czas oczekiwania (sumaryczny): {mean_waiting:.2f} s\n")
        f.write(f"Średnia prędkość pojazdów: {mean_speed:.2f} m/s\n")
        f.write(f"Średni czas oczekiwania pojedynczego pojazdu: {avg_waiting_per_vehicle:.2f} s\n")
    print(f"📄 Dane zapisane do pliku: {output_path}")

    # === Wykres ===
    plot_metrics(avg_waiting_times, total_halted_vehicles, mean_waiting, mean_speed, avg_waiting_per_vehicle, vehicle_speeds)

# === Wykres ===
def plot_metrics(waiting_times, halted_vehicles, mean_waiting, mean_speed, avg_waiting_per_vehicle, vehicle_speeds):
    steps = list(range(len(waiting_times)))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Lewa oś Y - Czas oczekiwania
    ax1.set_xlabel("Krok symulacji")
    ax1.set_ylabel("Czas oczekiwania (s)", color='tab:blue')
    ax1.plot(steps, waiting_times, color='tab:blue', label="Czas oczekiwania")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)

    # Prawa oś Y - Zatrzymane pojazdy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Zatrzymane pojazdy", color='tab:orange')
    ax2.plot(steps, halted_vehicles, color='tab:orange', label="Zatrzymane pojazdy")
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Trzecia oś Y - Prędkość pojazdów
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    ax3.set_frame_on(True)
    ax3.patch.set_visible(False)
    for sp in ax3.spines.values():
        sp.set_visible(True)
    ax3.set_ylabel("Średnia prędkość (m/s)", color='tab:green')
    ax3.plot(steps, vehicle_speeds, color='tab:green', label="Śr. prędkość")
    ax3.tick_params(axis='y', labelcolor='tab:green')

    # Statystyki tekstowe na wykresie
    stats_text = (
        f"Śr. czas oczekiwania: {mean_waiting:.2f} s\n"
        f"Śr. prędkość pojazdów: {mean_speed:.2f} m/s\n"
        f"Śr. czas na pojazd: {avg_waiting_per_vehicle:.2f} s"
    )
    plt.title("Czas oczekiwania, zatrzymane pojazdy i prędkość")
    plt.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.7))

    fig.tight_layout()
    plt.show()

# === Start ===
if __name__ == "__main__":
    test_model()
