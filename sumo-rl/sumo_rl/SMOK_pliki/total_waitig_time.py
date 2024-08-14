import traci

try:
    traci.close()
except traci.exceptions.FatalTraCIError:
    pass  # Ignoruj błąd, jeśli połączenie nie było aktywne

# Uruchomienie SUMO z plikiem konfiguracji
traci.start(["sumo-gui", "-c", "c:\\DATA\\ROB\\7SEM\\test\\praca_In\\sumo-rl\\sumo_rl\\nets\\single-intersection\\single-intersection.sumocfg"])

total_waiting_time = 0
simulation_duration = 200  # Czas symulacji w sekundach

while traci.simulation.getTime() < simulation_duration:
    traci.simulationStep()
    vehicles = traci.vehicle.getIDList()
    for vehicle_id in vehicles:
        total_waiting_time += traci.vehicle.getWaitingTime(vehicle_id)

traci.close()

print(f"Total waiting time: {total_waiting_time} seconds")
