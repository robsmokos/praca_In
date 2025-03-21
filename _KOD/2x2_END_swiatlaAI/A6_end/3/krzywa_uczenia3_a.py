import matplotlib.pyplot as plt
import numpy as np

# 📌 Dane: numery epizodów i całkowite nagrody
epizody = np.arange(68)  # 70 epizodów (od 0 do 69)
nagrody = np.array([
            3461.08, -6206.11, -6778.50, -1052.79, 1505.98, -2101.82, 2960.91, 
           -1941.36, -2466.54, -2830.19, 2379.13, 833.43, 2525.88, -1155.55, 
           2885.72, -5980.51, -9796.71, 2735.02, 3982.40, 4238.67, -1108.09, 
           -2981.20, 2612.27, 2975.24, 3135.78, -1912.62, 4148.71, 4229.66, 
           -4076.61, 191.60, 3223.76, -10815.38, -8026.96, -9947.20, 291.57, 
           -7093.04, -2465.20, 112.59, -3018.17, -1140.82, 1928.80, 2338.57, 
           2909.61, 2112.80, 1954.93, -3581.82, -14580.28, -723.68, 2594.59, 
           52.86, 3910.33, -406.38, 2239.25, 961.37, 4375.72, 4615.88, 3824.13, 
           3093.17, 592.32, -1390.39, 1604.37, 2194.93, 3356.96, -4852.13, 
           1049.82, 1468.47, 1940.97, -1517.97
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
plt.plot(epizody[window_size-1:], srednia_ruchoma, linestyle=':', color='r', linewidth=2, label=f"Średnia")

# 📌 Oznaczenia wykresu
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # Linia pozioma dla nagrody 0
plt.xlabel("Epizod")
plt.ylabel("Całkowita nagroda")
plt.title("Nagrody w epizodach")
plt.legend()
plt.grid(True)
plt.show()
