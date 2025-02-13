import os
import traci

sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")  # sumo-gui.exe dla wersji z GUI

# Plik konfiguracji SUMO (zmień na swoją ścieżkę)
sumo_config = "2x2.sumocfg"  # Zmień na własną ścieżkę!


# Uruchomienie symulacji
traci.start([sumo_binary, "-c", sumo_config, '--start','--quit-on-end'])


print("Symulacja SUMO uruchomiona.")

step = 0
while step < 4000:  # Symulacja przez 100 kroków
    traci.simulationStep()  # Wykonaj krok symulacji
    print(f"Krok symulacji: {step}")

    # Pobieranie informacji o pojazdach w symulacji
    vehicles = traci.vehicle.getIDList()
    print(f"Liczba pojazdów w ruchu: {len(vehicles)}")

    step += 1

traci.close()
print("Symulacja zakończona.")    