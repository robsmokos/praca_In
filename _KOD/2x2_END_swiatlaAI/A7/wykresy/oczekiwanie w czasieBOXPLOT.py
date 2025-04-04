import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytywanie danych z plików
df_ai = pd.read_csv("test_modelu_AI_01.csv")
df_optymalne = pd.read_csv("swiatla_optymalne.csv")
df_glupie = pd.read_csv("sw_sekwencyjne.csv")

# Dodanie kolumny "System" dla identyfikacji danych
df_ai["System"] = "Algorytm Aktor-Krytyk"
df_optymalne["System"] = "Algorytm SUMO"
df_glupie["System"] = "Algorytm stałoczasowy"

# Połączenie danych do jednego DataFrame
df_all = pd.concat([
    df_ai[["CzasOczekiwania(s)", "System"]],
    df_optymalne[["CzasOczekiwania(s)", "System"]],
    df_glupie[["CzasOczekiwania(s)", "System"]]
], ignore_index=True)

# Wykres pudełkowy (boxplot)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x="System", y="CzasOczekiwania(s)", palette=["#fc5a50", "#25a36f", "#6488ea"])
#plt.title("Porównanie rozkładu czasów oczekiwania w różnych systemach")
plt.xlabel("System sterowania")
plt.ylabel("Czas oczekiwania (s)")
plt.grid(True)
plt.tight_layout()
plt.show()
