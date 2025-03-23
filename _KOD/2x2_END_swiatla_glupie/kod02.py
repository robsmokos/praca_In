import os
import traci
import pandas as pd

# Ścieżka do SUMO
sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui.exe")  # lub "sumo.exe" bez GUI

# Plik konfiguracyjny SUMO
sumo_config = "2x2.sumocfg"  # <- zmień na swoją ścieżkę

# Uruchomienie SUMO
traci.start([sumo_binary, "-c", sumo_config, "--start", "--quit-on-end"])

print("Symulacja SUMO uruchomiona.")

# Lista do przechowywania danych
dane = []

# Liczba kroków symulacji
max_steps = 5800
step = 0

while step < max_steps:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()
    num_vehicles = len(vehicles)
    
    total_wait_time = 0.0
    num_stopped = 0
    total_speed = 0.0

    for veh_id in vehicles:
        wait_time = traci.vehicle.getWaitingTime(veh_id)
        speed = traci.vehicle.getSpeed(veh_id)
        
        total_wait_time += wait_time
        total_speed += speed
        
        if speed < 0.1:  # Pojazd praktycznie stoi
            num_stopped += 1

    avg_speed = total_speed / num_vehicles if num_vehicles > 0 else 0.0

    # Zapisz dane z danego kroku
    dane.append({
        "Krok": step,
        "CzasOczekiwania(s)": round(total_wait_time, 2),
        "ZatrzymanePojazdy": num_stopped,
        "SredniaPredkosc(m/s)": round(avg_speed, 2)
    })

    step += 1

traci.close()
print("Symulacja zakończona.")

# Zapisz dane do pliku CSV
df = pd.DataFrame(dane)
df.to_csv("wyniki_symulacji.csv", index=False)
print("Dane zapisane do pliku 'wyniki_symulacji.csv'.")
