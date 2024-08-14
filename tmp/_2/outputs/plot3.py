import os
import time
import pandas as pd
import plotly.graph_objects as go

def update_plot(fig, directory, x_column, y_columns):
    while True:
        # Pobierz listę plików CSV w katalogu
        files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        
        # Wyświetl listę plików w katalogu
        print("Pliki CSV w katalogu:", files)
        
        # Wybierz najnowszy plik
        latest_file = max(files, key=os.path.getctime)
        print("Wybrany plik:", latest_file)  # Dodaj wydruk wybranego pliku
        
        # Wczytaj dane z pliku CSV
        df = pd.read_csv(os.path.join(directory, latest_file))

        # Aktualizacja danych na wykresie
        fig.data = []
        for y_column in y_columns:
            fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode="lines", name=y_column))

        # Oczekiwanie przed kolejną aktualizacją
        time.sleep(5)  # Odczekaj 5 sekund przed ponowną aktualizacją

# Wybór katalogu, w którym znajdują się pliki CSV
directory = ""

# Wybór kolumn do wyświetlenia na wykresie
x_column = "step"
y_columns = ["1_stopped", "2_stopped"]

# Utworzenie obiektu wykresu
fig = go.Figure()

# Konfiguracja układu i tytułu wykresu
fig.update_layout(
    title="Liczba zatrzymanych pojazdów",
    xaxis_title="Krok",
    yaxis_title="Liczba zatrzymanych pojazdów",
    legend_title="Kolumny",
)

# Wywołanie funkcji aktualizującej wykres
update_plot(fig, directory, x_column, y_columns)
