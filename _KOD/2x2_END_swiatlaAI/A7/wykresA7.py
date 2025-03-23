import matplotlib.pyplot as plt
import numpy as np

# 🔹 Dane: epizody i odpowiadające im nagrody
episodes = list(range(31))  # Numery epizodów od 0 do 60
rewards = [
-6281.77, -5125.88, -7777.81, -2176.37, -3903.43, -11671.11, 4812.29, 4237.20, 4652.62, 4781.03,
4811.17, 4247.13, 4252.29, 4711.27, 5088.17, 1247.44, 711.69, 4965.64, -49.17, 5009.64,
-991.74, 4980.92, 650.43, 4771.42, -1104.00, 4523.16, 1407.78, 1049.27, 4455.06, -45697.22,
3627.87

]
# 🔹 Obliczanie średniej kroczącej (okno = 5 epizodów)
window_size = 5
moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# 🔹 Rysowanie wykresu
plt.figure(figsize=(10, 5))
plt.plot(episodes, rewards,  linewidth=0.2, linestyle='-', color='b', label="Całkowita nagroda")
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
