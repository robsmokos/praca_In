import traci

try:
    traci.close()
except traci.exceptions.FatalTraCIError:
    pass  # Ignoruj błąd, jeśli połączenie nie było aktywne

# Uruchomienie SUMO z plikiem konfiguracji
traci.start(["sumo-gui", "-c", "c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\nets\\single-intersection\\single-intersection.sumocfg", "-S"])

total_waiting_time = 0
simulation_duration = 10000  # Czas symulacji w sekundach
sampling_interval = 5  # Częstotliwość próbkowania w sekundach

# Otwórz plik do zapisu
with open("test.txt", "w") as logfile:
    logfile.write("Czas(s),Calkowity czas oczekiwania(s)\n")  # Nagłówek pliku

    while traci.simulation.getTime() < simulation_duration:
        traci.simulationStep()

        # Próbkowanie co 5 sekund
        if int(traci.simulation.getTime()) % sampling_interval == 0:
            vehicles = traci.vehicle.getIDList()
            system_total_waiting_time = sum(traci.vehicle.getWaitingTime(vehicle_id) for vehicle_id in vehicles)
            
            # Zapisz dane do pliku
            logfile.write(f"{traci.simulation.getTime()},{system_total_waiting_time}\n")

traci.close()
