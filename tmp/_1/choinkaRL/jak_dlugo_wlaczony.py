import pandas as pd

# Wczytaj dane z pliku CSV
df = pd.read_csv('traffic_light_actions.csv')

# Inicjalizacja nowej kolumny do przechowywania czasu trwania włączenia sygnalizatora
df['time_on_duration'] = None

# Słownik do śledzenia ostatniego czasu, gdy sygnalizator był włączony
last_on_time = {}

# Iteracja po DataFrame
for index, row in df.iterrows():
    traffic_light_id = row['traffic_light_id']
    action = row['action']
    current_time = row['time']

    if action == 0:
        # Oblicz czas, przez jaki sygnalizator był włączony
        if traffic_light_id in last_on_time:
            df.at[index, 'time_on_duration'] = current_time - last_on_time[traffic_light_id]
    elif action == 1:
        # Zapisz czas włączenia sygnalizatora
        last_on_time[traffic_light_id] = current_time

# Sprawdź ostatnie stany sygnalizatorów
for index, row in df.iterrows():
    traffic_light_id = row['traffic_light_id']
    action = row['action']
    
    # Jeżeli sygnalizator jest włączony na końcu pliku, wyłączamy go na potrzeby kalkulacji
    if action == 1 and traffic_light_id in last_on_time:
        df.at[index, 'time_on_duration'] = 0.0

# Zapisz wyniki do nowego pliku CSV
df.to_csv('out_jak_dlugo_wlaczone.csv', index=False)

print("Wyniki zostały zapisane do pliku 'out_jak_dlugo_wlaczone.csv'")
