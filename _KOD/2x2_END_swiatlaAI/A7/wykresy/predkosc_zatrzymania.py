import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane
ai_df = pd.read_csv("test_modelu_AI_01.csv")
seq_df = pd.read_csv("sw_sekwencyjne.csv")
opt_df = pd.read_csv("swiatla_optymalne.csv")

# Wykres z przenikającymi się kolorami (przezroczystość) dla analizy rozrzutu
plt.figure(figsize=(12, 6))

# Wykresy scatter z większą przezroczystością (alpha) dla lepszego przenikania
plt.scatter(ai_df["ZatrzymanePojazdy"], ai_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="AI", color="skyblue", edgecolors='k', linewidths=0.3)

plt.scatter(seq_df["ZatrzymanePojazdy"], seq_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="Sekwencyjne", color="orange", edgecolors='k', linewidths=0.3)

plt.scatter(opt_df["ZatrzymanePojazdy"], opt_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="Optymalne", color="lightgreen", edgecolors='k', linewidths=0.3)

# Opisy wykresu
plt.title("Analiza rozrzutu z przenikającymi się barwami")
plt.xlabel("Zatrzymane pojazdy")
plt.ylabel("Średnia prędkość (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Pokaż wykres
plt.show()
