import matplotlib.pyplot as plt
import numpy as np

# 🔹 Dane: epizody i odpowiadające im nagrody
episodes = list(range(65))  # Numery epizodów od 0 do 60
rewards = [
    -1425.29, -5093.52, -5426.55, 4161.96, 111.53, -10730.29, -4631.99, -10393.26, -8474.59, -1091.94,
    -4467.67, -9579.02, -10528.85, -11897.22, -10624.23, -8475.50, -9220.26, 1124.47, -8917.71, -1680.76, 
    1208.01, -3938.71, -2273.10, -1410.81, -7063.26, -6007.17, -9874.59, -9482.74, -3621.48, -9612.41, 
    325.18, -1468.49, 2365.21, -690.44, 3322.08, -503.83, 4343.28, 3581.57, -75.53, 335.57, 2941.11, 
    -11442.01, -10033.57, -342.06, 4005.19, 1175.88, 4384.29, 4637.39, 2898.96, -2847.27, 1361.04, 
    -9541.93, 1992.84, 2597.10, 3581.25, 472.28, -5167.72, -8560.99, 4670.58, 2801.88, 2433.09, 2646.43, 
    4219.17, 4101.67, 3180.38
]

# 🔹 Obliczanie średniej kroczącej (okno = 5 epizodów)
window_size = 5
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# 🔹 Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, marker='o', linewidth=1, linestyle='-', color='b', label="Całkowita nagroda")
plt.plot(episodes[window_size-1:], moving_avg, linestyle=':', color='r', linewidth=2, label=f"Średnia")
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Linia y=0 dla odniesienia

# 🔹 Opisy osi
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.title("Nagroda w epizodach")
plt.legend()
plt.grid(True)

# 🔹 Wyświetlenie wykresu
plt.show()
