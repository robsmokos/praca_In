import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z plików
df_ai = pd.read_csv("test_modelu_AI_01.csv")
df_optymalne = pd.read_csv("swiatla_optymalne.csv")
df_sekwencyjne = pd.read_csv("sw_sekwencyjne.csv")

# Wyliczenie liczby sekwencji (gdzie czas oczekiwania > 0)
def calculate_sequences(df):
    return (df["CzasOczekiwania(s)"] > 0).cumsum()

# Dodanie kolumn z liczbą sekwencji
df_ai["Sekwencja"] = calculate_sequences(df_ai)
df_optymalne["Sekwencja"] = calculate_sequences(df_optymalne)
df_sekwencyjne["Sekwencja"] = calculate_sequences(df_sekwencyjne)

# Obliczenie całkowitej liczby sekwencji potrzebnych w każdej strategii
total_sequences = {
    "Aktor-Krytyk AI": df_ai["Sekwencja"].iloc[-1],
    "optymalizator SUMO": df_optymalne["Sekwencja"].iloc[-1],
    "sterowanie sekwencyjne": df_sekwencyjne["Sekwencja"].iloc[-1]
}

# Tworzenie wykresu słupkowego
plt.figure(figsize=(8, 6))
plt.bar(total_sequences.keys(), total_sequences.values(), color=["#fc5a50", "#25a36f", "#6488ea" ])
plt.title("Łączna liczba sekwencji potrzebna do opuszczenia skrzyżowania")
plt.ylabel("Liczba sekwencji")
plt.tight_layout()
plt.show()
