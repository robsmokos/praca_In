import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z plików
ai_df = pd.read_csv('/mnt/data/AI.txt', sep='\t')
glupie_df = pd.read_csv('/mnt/data/swiatla_glupie.txt', sep=';', comment='#')
optymalne_df = pd.read_csv('/mnt/data/swiatlaOptymalne.txt', sep=';', comment='#')

# Usunięcie spacji z nazw kolumn
ai_df.columns = ai_df.columns.str.strip()
glupie_df.columns = glupie_df.columns.str.strip()
optymalne_df.columns = optymalne_df.columns.str.strip()

# Tworzenie wykresu
plt.figure(figsize=(14, 6))
plt.plot(ai_df['Krok'], ai_df['SredniaPredkosc(m/s)'], label='AI', linewidth=2)
plt.plot(glupie_df['Time'], glupie_df['avg. speed [m/s]'], label='Światła Głupie', linewidth=2)
plt.plot(optymalne_df['Time'], optymalne_df['avg. speed [m/s]'], label='Światła Optymalne', linewidth=2)

plt.title('Średnia prędkość w czasie dla różnych scenariuszy')
plt.xlabel('Czas [s]')
plt.ylabel('Średnia prędkość [m/s]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
