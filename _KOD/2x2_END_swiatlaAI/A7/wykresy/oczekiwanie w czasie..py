import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane z plików CSV
file_ai = pd.read_csv("test_modelu_AI_01.csv")
file_opt = pd.read_csv("swiatla_optymalne.csv")
file_seq = pd.read_csv("sw_sekwencyjne.csv")

# Tworzenie wykresu: czas oczekiwania w czasie (krok symulacji)
plt.figure(figsize=(14, 6))

plt.plot(file_ai["Krok"], file_ai["CzasOczekiwania(s)"], label="Model AI", alpha=0.8, color="#fc5a50")
plt.plot(file_opt["Krok"], file_opt["CzasOczekiwania(s)"], label="Algorytm SUMO", alpha=0.8, color="#25a36f")
plt.plot(file_seq["Krok"], file_seq["CzasOczekiwania(s)"], label="Algorytm stałoczasowy", alpha=0.8, color="#6488ea")

# Opis osi i wykresu
plt.xlabel("Krok symulacji")
plt.ylabel("Czas oczekiwania (s)")
#plt.title("Porównanie czasu oczekiwania pojazdów w czasie")
plt.legend()
plt.grid(linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()
