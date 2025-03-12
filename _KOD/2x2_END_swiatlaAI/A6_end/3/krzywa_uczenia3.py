import matplotlib.pyplot as plt
import numpy as np

# 📌 Dane: numery epizodów i całkowite nagrody
epizody = np.arange(70)  # 70 epizodów (od 0 do 69)
nagrody = np.array([
    -501.67, -5395.62, -10339.80, -7155.89, -1235.59, -7031.86, -5408.12, -4854.07, 
    464.86, -6478.20, -5813.37, 4191.67, np.nan, -4844.53, -9241.51, -691.19, 
    -2356.27, -8240.08, -2781.40, -4535.22, -3207.60, -192.27, -4219.77, 
    528.25, -9709.00, -6036.71, -9701.22, -3142.09, -7311.79, -4177.59, 
    130.31, -5420.81, 2508.91, -3541.30, -8167.81, -9237.50, -782.51, 
    -2190.51, 3226.94, -1459.98, -4213.82, -45.82, 301.71, -3243.16, 1424.40, 
    -6577.16, -5872.28, -4598.56, 2999.50, 3320.26, 391.79, -1867.00, 2476.50, 
    -2869.00, -2811.86, 107.61, -3636.69, -2231.96, -10061.57, -4056.09, 302.33,
    -7377.61, 710.04, -5611.81, 3395.99, 703.75, -8026.46, -7940.11, 2242.24, 2196.64
])

# 📌 Usunięcie wartości NaN poprzez interpolację prostym sposobem
for i in range(1, len(nagrody) - 1):
    if np.isnan(nagrody[i]):
        nagrody[i] = (nagrody[i - 1] + nagrody[i + 1]) / 2

# 📌 Obliczenie średniej ruchomej (okno = 5 epizodów)
window_size = 5
srednia_ruchoma = np.convolve(nagrody, np.ones(window_size)/window_size, mode='valid')

# 📌 Wykres nagród w epizodach
plt.figure(figsize=(12, 6))
plt.plot(epizody, nagrody, marker='o', linestyle='-', color='b', alpha=0.5, label="Nagroda w epizodzie")

# 📌 Wykres średniej ruchomej (mniej punktów, ponieważ convolution skraca listę)
plt.plot(epizody[window_size-1:], srednia_ruchoma, linestyle='-', color='r', linewidth=2, label=f"Średnia ruchoma (okno={window_size})")

# 📌 Oznaczenia wykresu
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # Linia pozioma dla nagrody 0
plt.xlabel("Epizod")
plt.ylabel("Całkowita nagroda")
plt.title("Nagrody w epizodach i średnia ruchoma")
plt.legend()
plt.grid(True)
plt.show()
