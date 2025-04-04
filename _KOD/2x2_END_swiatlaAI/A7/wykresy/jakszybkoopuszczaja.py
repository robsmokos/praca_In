import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z plików CSV
df_ai = pd.read_csv("test_modelu_AI_01.csv")
df_optymalne = pd.read_csv("swiatla_optymalne.csv")
df_sekwencyjne = pd.read_csv("sw_sekwencyjne.csv")

# Funkcja do obliczenia liczby sekwencji
def calculate_sequences(df):
    return (df["CzasOczekiwania(s)"] > 0).cumsum()

# Dodanie kolumny 'Sekwencja'
df_ai["Sekwencja"] = calculate_sequences(df_ai)
df_optymalne["Sekwencja"] = calculate_sequences(df_optymalne)
df_sekwencyjne["Sekwencja"] = calculate_sequences(df_sekwencyjne)

# Obliczenie całkowitej liczby sekwencji
total_sequences = {
    "Algorytm Aktor-Krytyk": df_ai["Sekwencja"].iloc[-1],
    "Algorytm SUMO": df_optymalne["Sekwencja"].iloc[-1],
    "Algorytm stałoczasowy": df_sekwencyjne["Sekwencja"].iloc[-1]
}

# Tworzenie wykresu słupkowego z dodatkowymi usprawnieniami
fig, ax = plt.subplots(figsize=(8, 2.5))  # maksymalnie niski wykres
bars = ax.bar(total_sequences.keys(), total_sequences.values(), color=["#fc5a50", "#25a36f", "#6488ea"])

# Dodanie wartości nad słupkami
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0, f'{int(height)}', ha='center', va='bottom', fontsize=9)

# Dostosowanie skali osi Y dla lepszej widoczności
min_val = min(total_sequences.values())
max_val = max(total_sequences.values())
ax.set_ylim(min_val * 0.98, max_val * 1.02)

# Dodanie tytułu, etykiety osi Y oraz siatki
#ax.set_title("Łączna liczba sekwencji potrzebna do opuszczenia skrzyżowania")
ax.set_ylabel("Liczba sekwencji")
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
