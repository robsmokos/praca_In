import os
import sys
import csv

# Sprawdzenie, czy zmienna środowiskowa SUMO_HOME jest ustawiona
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Proszę zadeklarować zmienną środowiskową 'SUMO_HOME'")

# Importowanie niezbędnych modułów z biblioteki sumo-rl oraz traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
import traci  # Dodanie traci do obsługi emisji

if __name__ == "__main__":
    # Parametry algorytmu Q-learning
    alpha = 0.1  # Współczynnik uczenia
    gamma = 0.99  # Współczynnik dyskontowania
    decay = 1  # Współczynnik zmniejszania epsilon
    runs = 1  # Liczba uruchomień symulacji
    episodes = 1  # Liczba epizodów w każdej symulacji

    # Inicjalizacja środowiska SUMO
    env = SumoEnvironment(
        net_file="test_flow.net.xml",  # Plik z siecią drogową
        route_file="test_flow.rou.xml",  # Plik z trasami pojazdów
        use_gui=False,  # Wyłączenie interfejsu graficznego
        num_seconds=10000,  # Czas trwania symulacji w sekundach
        min_green=5,  # Minimalny czas trwania zielonego światła
        delta_time=5,  # Czas między kolejnymi decyzjami agentów
    )

    # Przygotowanie plików CSV do zapisu wyników
    csv_file_path = 'a1_out__traffic_light_actions.csv'  # Plik do zapisu działań sygnalizacji świetlnej
    emissions_csv_file_path = 'a1_out__vehicle_emissions.csv'  # Plik CSV do zapisu emisji spalin

    # Otwarcie plików CSV do zapisu
    with open(csv_file_path, mode='w', newline='') as file, open(emissions_csv_file_path, mode='w', newline='') as emissions_file:
        writer = csv.writer(file)
        emissions_writer = csv.writer(emissions_file)
        
        # Zapis nagłówków do plików CSV
        writer.writerow(['run', 'time', 'traffic_light_id', 'action'])
        emissions_writer.writerow(['run', 'time', 'vehicle_id', 'CO2', 'CO', 'HC', 'NOx', 'PMx'])

        for run in range(1, runs + 1):
            # Resetowanie środowiska i inicjalizacja agentów
            initial_states = env.reset()
            ql_agents = {
                ts: QLAgent(
                    starting_state=env.encode(initial_states[ts], ts),
                    state_space=env.observation_space,
                    action_space=env.action_space,
                    alpha=alpha,
                    gamma=gamma,
                    exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
                )
                for ts in env.ts_ids
            }

            for episode in range(1, episodes + 1):
                if episode != 1:
                    initial_states = env.reset()
                    for ts in initial_states.keys():
                        ql_agents[ts].state = env.encode(initial_states[ts], ts)

                done = {"__all__": False}
                while not done["__all__"]:
                    # Agenci wykonują swoje działania
                    actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                    # Zapis działań sygnalizacji świetlnej do pliku CSV
                    current_time = env.sim_step  # Pobieranie aktualnego czasu symulacji
                    for ts, action in actions.items():
                        writer.writerow([run, current_time, ts, action])

                    # Pobieranie danych o emisjach spalin z symulacji
                    for veh_id in traci.vehicle.getIDList():
                        co2 = traci.vehicle.getCO2Emission(veh_id)
                        co = traci.vehicle.getCOEmission(veh_id)
                        hc = traci.vehicle.getHCEmission(veh_id)
                        nox = traci.vehicle.getNOxEmission(veh_id)
                        pmx = traci.vehicle.getPMxEmission(veh_id)
                        emissions_writer.writerow([run, current_time, veh_id, co2, co, hc, nox, pmx])

                    # Wykonanie kroku w symulacji
                    s, r, done, info = env.step(action=actions)

                    # Uczenie agentów na podstawie nowych stanów i nagród
                    for agent_id in s.keys():
                        ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

                # Zapis wyników symulacji do pliku CSV
                env.save_csv(f"choinka_rl{run}", episode)

    env.close()  # Zamknięcie środowiska SUMO
    traci.close()  # Zamknięcie traci po zakończeniu symulacji
