# SMOTER wylicza redni czas oczekiwania pojazdu dla każdej epoki
#
#

import os
import pandas as pd

# Ścieżka do katalogu bieżącego
directory = os.getcwd()

# Lista wszystkich plików w katalogu z rozszerzeniem .csv
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Nazwa pliku wynikowego
output_file = 'results.txt'

# Otwarcie pliku wynikowego do zapisu
with open(output_file, 'w') as result_file:
    # Przetwarzanie każdego pliku CSV
    for file_name in csv_files:
        # Pełna ścieżka do pliku
        file_path = os.path.join(directory, file_name)
        
        # Wczytanie danych z pliku CSV
        data = pd.read_csv(file_path)
        
        # Obliczenie sumy wartości w kolumnie "system_total_waiting_time"
        total_waiting_time = data['system_total_waiting_time'].sum()
        
        # Obliczenie sumy wartości w kolumnie "system_total_stopped"
        total_stopped = data['system_total_stopped'].sum()
        
        # Obliczenie średniej długości zatrzymania pojazdów
        if total_stopped != 0:
            average_stop_duration = total_waiting_time / total_stopped
        else:
            average_stop_duration = 0
        
        # Zapisanie wyniku dla bieżącego pliku do pliku tekstowego
        result_file.write(f'Plik: {file_name} - Średnia długość zatrzymania pojazdów: {average_stop_duration:.2f} jednostek czasu\n')
