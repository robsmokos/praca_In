import os
import time
import pandas as pd
import plotly.graph_objects as go

def update_plot(fig, directory, x_column, y_columns):
    while True:
        # Pobierz listę plików CSV w katalogu
        files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        
        # Wybierz najnowszy plik
        latest_file = max(files, key=os.path.getctime)
        
        # Wczytaj dane z pliku CSV
        df = pd.read_csv(os.path.join(directory, latest_file))

        # Aktualizacja danych na wykresie
        fig.data = []
        for y_column in y_columns:
            fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], mode="lines", name=y_column))

        # Zapisz wykres jako plik HTML
        fig.write_html("wykres.html")

        # Oczekiwanie przed kolejną aktualizacją
        time.sleep(5)  # Odczekaj 5 sekund przed ponowną aktualizacją

# Ustalenie ścieżki katalogu, w którym znajdują się pliki CSV
directory = "c:\\DATA\\ROB\\6SEM\\PRACA_DYP\\_2\\outputs\\single-intersection"

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
