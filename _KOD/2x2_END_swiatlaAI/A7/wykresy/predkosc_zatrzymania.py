import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane
ai_df = pd.read_csv("test_modelu_AI_01.csv")
seq_df = pd.read_csv("sw_sekwencyjne.csv")
opt_df = pd.read_csv("swiatla_optymalne.csv")

plt.figure(figsize=(12, 6))

# Najpierw rysujemy tło (z mniejszym zorder)
plt.scatter(seq_df["ZatrzymanePojazdy"], seq_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="sterowanie sekwencyjne", color="#6488ea", linewidths=0.3, zorder=1)

plt.scatter(opt_df["ZatrzymanePojazdy"], opt_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="optymalizator SUMO", color="#25a36f", linewidths=0.3, zorder=2)

# Na końcu czerwony – najwyższy zorder
plt.scatter(ai_df["ZatrzymanePojazdy"], ai_df["SredniaPredkosc(m/s)"], 
            alpha=0.2, label="Aktor-Krytyk AI", color="#fc5a50", linewidths=0.3, zorder=3)

plt.title("Analiza rozrzutu")
plt.xlabel("Zatrzymane pojazdy")
plt.ylabel("Średnia prędkość (m/s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
