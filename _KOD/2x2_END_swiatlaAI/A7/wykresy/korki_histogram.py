import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wczytanie danych
ai_df = pd.read_csv("test_modelu_AI_01.csv")
seq_df = pd.read_csv("sw_sekwencyjne.csv")
opt_df = pd.read_csv("swiatla_optymalne.csv")

# Obliczenie wspólnego zakresu danych
min_val = min(ai_df["ZatrzymanePojazdy"].min(), 
              seq_df["ZatrzymanePojazdy"].min(), 
              opt_df["ZatrzymanePojazdy"].min())

max_val = max(ai_df["ZatrzymanePojazdy"].max(), 
              seq_df["ZatrzymanePojazdy"].max(), 
              opt_df["ZatrzymanePojazdy"].max())

# Tworzymy biny (20 szerokich przedziałów)
num_bins = 20
bins = np.linspace(min_val, max_val, num_bins + 1)

# Histogramy (liczba przypadków w każdym przedziale)
ai_hist, _ = np.histogram(ai_df["ZatrzymanePojazdy"].dropna(), bins=bins)
seq_hist, _ = np.histogram(seq_df["ZatrzymanePojazdy"].dropna(), bins=bins)
opt_hist, _ = np.histogram(opt_df["ZatrzymanePojazdy"].dropna(), bins=bins)

# Pozycje słupków
x = np.arange(num_bins)
bar_width = 0.25

# Etykiety przedziałów
bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(num_bins)]

# Rysowanie wykresu
plt.figure(figsize=(14, 6))
plt.bar(x - bar_width, ai_hist, width=bar_width, label="Algorytm Aktor-Krytyk", color="#fc5a50")
plt.bar(x, seq_hist, width=bar_width, label="Algorytm stałoczasowy", color="#6488ea")
plt.bar(x + bar_width, opt_hist, width=bar_width, label="Algorytm SUMO", color="#25a36f")

plt.xticks(x, bin_labels, rotation=45)
plt.xlabel("Zakres liczby zatrzymań")
plt.ylabel("Liczba wystąpień")
#plt.title("Rozkład zatrzymań – słupki obok siebie (bez obwódek)")
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
