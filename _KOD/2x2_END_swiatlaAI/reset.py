import os
import traci
import psutil  # Biblioteka do zamykania procesów

# Ścieżka do SUMO
sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")
sumo_config = "2x2.sumocfg"

def stop_traci():
    """Funkcja zamykająca aktywne połączenie TraCI i procesy SUMO."""
    if traci.isLoaded():
        print("Zamykanie połączenia TraCI...")
        traci.close()

    # Sprawdzenie i zamknięcie procesów SUMO
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        try:
            if "sumo" in proc.info['name'].lower():
                print(f"Zamykanie procesu SUMO: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    print("Połączenie TraCI i procesy SUMO zostały zatrzymane.")

# Wywołanie funkcji zatrzymania
stop_traci()
