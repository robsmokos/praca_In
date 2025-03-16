import matplotlib.pyplot as plt
import numpy as np

# 🔹 Numery epizodów
episodes = list(range(58))  # 58 epizodów (od 0 do 57)

# 🔹 Nagrody (wklejone z logów)
total_rewards = [
    -10086.036, -8576.587, -12878.182, -11208.496, -12619.710, -10171.453, 
    -11357.314, -10733.232, -10494.243, -11805.970, -13947.276, -11815.597,
    -9505.797, -9996.072, -11405.216, -10372.892, -9467.552, -8703.109, 
    370.335, 136.705, 2113.544, 2709.670, -200.431, 1143.517, -1099.755, 
    -2218.630, 1618.127, -552.904, -34.906, -230.208, 1339.751, 229.807, 
    475.048, -3968.040, 1061.659, 1226.744, 1145.821, 404.074, 21.073, 
    -2032.990, -2209.947, -2021.939, -275.624, -4925.933, -1452.390, 
    -3413.156, -3987.838, -6851.146, -2871.288, -941.300, -4500.450, 
    -4520.546, -987.903, -1551.315, -2711.037, -937.724, -653.526, 
    -3420.761
]

# 🔹 Obliczanie średniej ruchomej (5 epizodów)
window_size = 5
rolling_mean = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')

# 🔹 Rysowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(episodes, total_rewards, marker='o', linestyle='-', color='b', label="Całkowita nagroda")
plt.plot(episodes[window_size-1:], rolling_mean, linestyle=':', color='r', label="Średnia")
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Linia y=0 dla odniesienia

# 🔹 Opisy osi
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.title("Nagroda w epizodach")
plt.legend()
plt.grid()

# 🔹 Pokazanie wykresu
plt.show()
