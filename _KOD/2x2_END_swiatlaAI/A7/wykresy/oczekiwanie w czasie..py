import pandas as pd
import matplotlib.pyplot as plt

# Wczytywanie danych z plików
df_ai = pd.read_csv("test_modelu_AI_01.csv")
df_optymalne = pd.read_csv("swiatla_optymalne.csv")
df_glupie = pd.read_csv("sw_sekwencyjne.csv")

# Konwersja kolumny 'Krok' na liczby całkowite
df_ai["Krok"] = pd.to_numeric(df_ai["Krok"], errors="coerce").dropna().astype(int)
df_optymalne["Krok"] = pd.to_numeric(df_optymalne["Krok"], errors="coerce").dropna().astype(int)
df_glupie["Krok"] = pd.to_numeric(df_glupie["Krok"], errors="coerce").dropna().astype(int)

# Tworzenie wykresu liniowego czasów oczekiwania w czasie
plt.figure(figsize=(14, 6))
plt.plot(df_ai["Krok"], df_ai["CzasOczekiwania(s)"], label="AI", color="green", alpha=0.9)
plt.plot(df_optymalne["Krok"], df_optymalne["CzasOczekiwania(s)"], label="Optymalne Światła", color="blue", alpha=0.9)
plt.plot(df_glupie["Krok"], df_glupie["CzasOczekiwania(s)"], label="Równe Sekwencje (Głupie)", color="red", alpha=0.9)

# Ustawienia wykresu
plt.xlabel("Krok (czas)")
plt.ylabel("Czas oczekiwania (s)")
plt.title("Zmiana czasów oczekiwania w czasie")
max_krok = max(df_ai["Krok"].max(), df_optymalne["Krok"].max(), df_glupie["Krok"].max())
plt.xticks(ticks=range(0, max_krok + 1, 500))
plt.legend()
plt.tight_layout()
plt.show()
