import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie danych z plików
df_ai = pd.read_csv("test_modelu_AI_01.csv")
df_optymalne = pd.read_csv("swiatla_optymalne.csv")
df_glupie = pd.read_csv("sw_sekwencyjne.csv")

# Konwersja kolumny 'Krok' na typ liczbowy
df_ai["Krok"] = pd.to_numeric(df_ai["Krok"], errors="coerce").dropna().astype(int)
df_optymalne["Krok"] = pd.to_numeric(df_optymalne["Krok"], errors="coerce").dropna().astype(int)
df_glupie["Krok"] = pd.to_numeric(df_glupie["Krok"], errors="coerce").dropna().astype(int)

# Tworzenie wykresu
plt.figure(figsize=(14, 6))
plt.plot(df_ai["Krok"], df_ai["SredniaPredkosc(m/s)"], label="Algorytm Aktor-Krytyk", color="#fc5a50", alpha=0.9)
plt.plot(df_optymalne["Krok"], df_optymalne["SredniaPredkosc(m/s)"], label="Algorytm SUMO", color="#25a36f", alpha=0.9)
plt.plot(df_glupie["Krok"], df_glupie["SredniaPredkosc(m/s)"], label="Algorytm  stałoczasowy", color="#6488ea", alpha=0.9)

# Oś X co 500 kroków
max_krok = max(df_ai["Krok"].max(), df_optymalne["Krok"].max(), df_glupie["Krok"].max())
plt.xticks(ticks=range(0, max_krok + 1, 500))

plt.xlabel("Krok (czas)")
plt.ylabel("Średnia Prędkość (m/s)")
#plt.title("Porównanie zmian średniej prędkości w czasie")
plt.legend()
plt.tight_layout()
plt.show()
