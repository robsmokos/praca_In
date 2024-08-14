import os
import pandas as pd
import re

# Ścieżka do katalogu bieżącego
directory = os.getcwd()

# Lista wszystkich plików w katalogu z rozszerzeniem .csv
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Funkcja do ekstrakcji numeru epoki z nazwy pliku
def extract_epoch_number(file_name):
    match = re.search(r'ep(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

# Sortowanie plików według numeru epoki
sorted_csv_files = sorted(csv_files, key=extract_epoch_number)

# Nazwa pliku wynikowego
output_file = 'results.txt'

# Otwarcie pliku wynikowego do zapisu
with open(output_file, 'w') as result_file:
    # Przetwarzanie każdego pliku CSV
    for file_name in sorted_csv_files:
        # Pełna ścieżka do pliku
        file_path = os.path.join(directory, file_name)
        
        try:
            # Wczytanie danych z pliku CSV
            data = pd.read_csv(file_path)
            
            # Sprawdzenie, czy wymagane kolumny istnieją w danych
            if 'system_total_waiting_time' in data.columns and 'system_total_stopped' in data.columns:
                # Obliczenie sumy wartości w kolumnie "system_total_waiting_time"
                total_waiting_time = data['system_total_waiting_time'].sum()
                
                # Obliczenie sumy wartości w kolumnie "system_total_stopped"
                total_stopped = data['system_total_stopped'].sum()
                
                # Obliczenie średniej długości zatrzymania pojazdów
                average_stop_duration = (total_waiting_time / total_stopped) if total_stopped != 0 else 0
                
                # Zapisanie wyniku dla bieżącego pliku do pliku tekstowego
                result_file.write(f'Plik: {file_name}\n')
                result_file.write(f'  Suma czasu oczekiwania: {total_waiting_time:.2f}\n')
                result_file.write(f'  Suma zatrzymań: {total_stopped}\n')
                result_file.write(f'  Średnia długość zatrzymania pojazdów: {average_stop_duration:.2f} jednostek czasu\n\n')
                
            else:
                result_file.write(f'Plik: {file_name} - Brak wymaganych kolumn\n\n')
        
        except Exception as e:
            result_file.write(f'Plik: {file_name} - Błąd: {str(e)}\n\n')
