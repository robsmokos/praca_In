import pandas as pd
import matplotlib.pyplot as plt

# Wczytaj dane z pliku (upewnij się, że AI.txt jest w tym samym folderze co ten skrypt)
df_ai = pd.read_csv("test_modelu_AI_01.csv")

# Konwersja kolumny "Krok" na typ liczbowy, jeśli nie jest
df_ai["Krok"] = pd.to_numeric(df_ai["Krok"], errors="coerce")

# Próbkowanie co 10. wiersz dla czytelności
df_sampled = df_ai.iloc[::10, :]

# Tworzenie wykresu z 3 osiami Y
fig, ax1 = plt.subplots(figsize=(14, 6))

# Oś lewa - Średnia prędkość
ax1.set_xlabel("Krok")
ax1.set_ylabel("Średnia prędkość (m/s)", color="royalblue")
ln1 = ax1.plot(df_sampled["Krok"], df_sampled["SredniaPredkosc(m/s)"], color="royalblue", label="Średnia prędkość (m/s)")
ax1.tick_params(axis='y', labelcolor="royalblue")

# Oś prawa - Zatrzymane pojazdy
ax2 = ax1.twinx()
ax2.set_ylabel("Zatrzymane pojazdy", color="orange")
ln2 = ax2.plot(df_sampled["Krok"], df_sampled["ZatrzymanePojazdy"], color="orange", label="Zatrzymane pojazdy")
ax2.tick_params(axis='y', labelcolor="darkorange")

# Druga oś prawa - Czas oczekiwania
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("axes", 1.1))  # przesunięcie drugiej osi
ax3.set_ylabel("Czas oczekiwania (s)", color="seagreen")
ln3 = ax3.plot(df_sampled["Krok"], df_sampled["CzasOczekiwania(s)"], color="seagreen", label="Czas oczekiwania (s)")
ax3.tick_params(axis='y', labelcolor="seagreen")

# Dodanie legendy
lns = ln1 + ln2 + ln3
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper left')

# Styl siatki
ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5)

# Tytuł i układ
plt.title("Wskaźniki sterowania AI w czasie (3 skale)")
plt.tight_layout()
plt.show()
