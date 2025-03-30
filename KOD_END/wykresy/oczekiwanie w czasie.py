import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane z plików CSV
file_ai = pd.read_csv("test_modelu_AI_01.csv")
file_opt = pd.read_csv("swiatla_optymalne.csv")
file_seq = pd.read_csv("sw_sekwencyjne.csv")

# Tworzenie wykresu: czas oczekiwania w czasie (krok symulacji)
plt.figure(figsize=(14, 6))

plt.plot(file_ai["Krok"], file_ai["CzasOczekiwania(s)"], label="Model AI", alpha=0.8)
plt.plot(file_opt["Krok"], file_opt["CzasOczekiwania(s)"], label="Światła optymalne", alpha=0.8)
plt.plot(file_seq["Krok"], file_seq["CzasOczekiwania(s)"], label="Światła sekwencyjne", alpha=0.8)

# Opis osi i wykresu
plt.xlabel("Krok symulacji")
plt.ylabel("Czas oczekiwania (s)")
plt.title("Porównanie czasu oczekiwania pojazdów w czasie")
plt.legend()
plt.grid(linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
