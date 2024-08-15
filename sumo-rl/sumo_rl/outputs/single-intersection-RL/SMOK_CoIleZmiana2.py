import csv

# Ścieżka do pliku CSV wejściowego i pliku wyjściowego
csv_file = 'traffic_light_actions.csv'  # Zastąp ścieżką do swojego pliku CSV
output_file = 'CoIleZmiana2.csv'  # Zastąp ścieżką do pliku wyjściowego

# Funkcja do odczytu danych, identyfikacji okresów zmiany świateł oraz zapisu wyników do pliku
def detect_signal_changes_with_duration(csv_file, output_file):
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        
        previous_action = None
        previous_time = None
        start_time = None
        
        # Otwarcie pliku wyjściowego do zapisu
        with open(output_file, mode='w') as output:
            for row in reader:
                current_time = float(row['time'])
                current_action = int(row['action'])
                
                # Ustawienie startowego czasu dla pierwszego stanu
                if start_time is None:
                    start_time = current_time
                
                # Jeśli nastąpiła zmiana akcji
                if previous_action is not None and current_action != previous_action:
                    duration = current_time - start_time
                    output.write(f"{duration:.1f} \n")
                    
                    # Zaktualizowanie startowego czasu na początek nowego stanu
                    start_time = current_time
                
                # Aktualizacja poprzednich wartości
                previous_action = current_action
                previous_time = current_time

# Wywołanie funkcji z zapisem do pliku
detect_signal_changes_with_duration(csv_file, output_file)
