import matplotlib.pyplot as plt
import numpy as np

# 🔹 Dane: epizody i odpowiadające im nagrody
episodes = list(range(61))  # Numery epizodów od 0 do 60
rewards = [
    -2146.89, -6221.31, -944.95, -2148.20, -6714.53, -11910.54, -10839.57, -12548.21, -13335.51, -10655.09, 
    -11783.70, -9906.03, -12058.84, -10928.68, -10456.95, -6650.68, 448.92, 2011.38, -946.70, -1241.89, 
    1634.33, 1929.48, -970.93, -4384.22, 482.92, 1794.74, 2655.77, 1395.37, -2051.93, 2523.46, 
    1177.81, -1851.44, -1977.67, 167.29, 2843.94, 1969.26, 2489.84, 2707.11, 62.16, -1440.19, 
    3040.68, 1415.82, 1803.71, 2920.12, 1210.64, -1129.67, -1571.60, 3564.00, 1778.81, -5611.91, 
    2693.22, 3429.87, 3800.08, -866.37, -5745.87, 2949.82, 3039.15, 1863.21, -6800.37, 100.79, 2924.67
]

# 🔹 Obliczanie średniej kroczącej (okno = 5 epizodów)
window_size = 5
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# 🔹 Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards, marker='o', linestyle='-', color='b', label="Całkowita nagroda")
plt.plot(episodes[window_size-1:], moving_avg, linestyle='-', color='r', linewidth=2, label=f"Średnia krocząca ({window_size})")
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)  # Linia y=0 dla odniesienia

# 🔹 Opisy osi
plt.xlabel("Epizod")
plt.ylabel("Nagroda")
plt.title("Zmiana całkowitej nagrody w czasie (ze średnią kroczącą)")
plt.legend()
plt.grid(True)

# 🔹 Wyświetlenie wykresu
plt.show()
