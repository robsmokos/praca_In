import matplotlib.pyplot as plt
import numpy as np

# 🔹 Dane: epizody i odpowiadające im nagrody
episodes = list(range(101))  # Numery epizodów od 0 do 60
rewards = [
359.10, 28.38, 202.61, 353.95, 1797.49, 2730.99, 3080.31, 2011.25, 3199.68, 3218.96,
    2958.73, -73.02, -1649.86, 3457.42, 2805.94, 3692.13, 3535.43, 3697.02, 3404.95, 2547.54,
    3535.57, 1178.35, 710.78, 3805.85, 2964.07, 1034.21, 1807.74, 1349.35, -4048.02, 1527.72,
    -1207.63, 668.25, 2597.02, 654.95, -433.75, -1026.74, -5850.90, -5199.29, 199.99, -1427.04,
    -4452.54, -990.45, -1272.43, 1914.56, -1204.36, -1955.19, 1271.27, 3716.31, 687.01, 2852.50,
    3680.26, 1623.66, 3820.37, 781.36, 525.50, 530.88, 3339.83, 3766.71, 1762.27, 2535.73,
    3597.84, 3714.54, 3682.00, 3670.12, 3667.98, -813.30, 1581.71, 3632.14, 1615.21, 3753.56,
    3732.47, 3674.11, -379.17, 598.45, 3728.23, 3665.87, 3690.49, 2736.99, 3772.69, 3742.32,
    3787.77, 3705.51, 3738.98, 3781.66, 3577.52, 2627.96, 3744.19, 3804.28, -2324.53, 1828.48,
    3696.83, 2588.66, 3611.82, 2801.57, 1152.43, 3658.08, -2566.71, 3685.30, 911.62, 870.85,
    2321.34
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
