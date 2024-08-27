import os
import sys
import traci
import csv

# Sprawdzenie, czy zmienna środowiskowa SUMO_HOME jest ustawiona
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Proszę zadeklarować zmienną środowiskową 'SUMO_HOME'")



# Ścieżki do plików sieci i tras
net_file = "a_4__test_flow.net.xml"  # Plik z siecią drogową
route_file = "test_flow.rou.xml"  # Plik z trasami pojazdów

# Komenda do uruchomienia SUMO z interfejsem traci
sumo_cmd = [
    "sumo",  # Można zmienić na "sumo", jeśli nie potrzebujesz GUI
    "-n", net_file,
    "-r", route_file,
    "--start",
    "--step-length", "5.0"  # Długość kroku symulacji w sekundach
]

# Uruchomienie SUMO z traci
traci.start(sumo_cmd)

# Liczba kroków symulacji
simulation_steps = 10000

# Nazwa pliku do zapisu wyników
output_file = "waiting_time_log.csv"

# Tworzenie i otwieranie pliku CSV do zapisu
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Zapis nagłówków kolumn
    writer.writerow(["Czas(s)", "Calkowity czas oczekiwania(s)"])

    # Główna pętla symulacji
    for step in range(1, simulation_steps + 1):
        traci.simulationStep()  # Wykonanie kroku symulacji
        current_time = step * 5.0  # Ponieważ krok symulacji wynosi 5.0 sekund
        
        # Obliczanie system_total_waiting_time
        system_total_waiting_time = sum([traci.vehicle.getAccumulatedWaitingTime(veh_id) for veh_id in traci.vehicle.getIDList()])
        
        # Zapis danych do pliku
        writer.writerow([current_time, system_total_waiting_time])

# Zakończenie symulacji i zamknięcie traci
traci.close()
