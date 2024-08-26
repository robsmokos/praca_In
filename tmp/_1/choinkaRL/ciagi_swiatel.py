import pandas as pd

# Wczytaj dane z pliku CSV
input_file = 'out_jak_dlugo_wlaczone.csv'  # Zamień na nazwę swojego pliku wejściowego
df = pd.read_csv(input_file)

# Wybierz tylko dwie ostatnie kolumny
df_result = df[['traffic_light_id', 'time_on_duration']]

# Zapisz wyniki do nowego pliku CSV
output_file = 'out_ciag_swiatel.csv'  # Nazwa pliku wyjściowego
df_result.to_csv(output_file, index=False)

print(f"Wyniki zostały zapisane do pliku '{output_file}'")
