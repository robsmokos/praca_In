import matplotlib.pyplot as plt
import numpy as np

# 🔹 Dane: epizody i odpowiadające im nagrody
episodes = list(range(28))  # Numery epizodów od 0 do 60
rewards = [
   -6005.23, 4678.22, 5007.61, 5580.90, 5690.08, 5812.67, 5853.67, 5909.11, 5342.47, 
    5935.19, 5756.22, 5037.91, 5844.93, 5565.47, 5925.15, 382.43, 1018.77, 5831.47, 
    5923.52, 5867.30, 3355.99, 5910.10, 5902.62, 3614.91, 5878.61, 5925.51, 3813.00, 
    4836.41
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
