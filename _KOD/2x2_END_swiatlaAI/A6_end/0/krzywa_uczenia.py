import matplotlib.pyplot as plt
import numpy as np

# 🔹 Numery epizodów
episodes = list(range(73))  # 73 epizody

# 🔹 Nagrody (zaktualizowana lista)
total_rewards = [
    -3026.38, -7353.98, -10267.03, -8746.78, -1376.08, -4703.27, -7559.73, -6621.76, -8115.88, 
    -11414.21, -8125.41, -10711.82, -6953.66, -6969.04, -5909.32, -8256.86, -2909.96, -3197.60,
    -1871.91, -4983.17, -2130.11, -4295.61, -2477.18, 312.95, -4220.36, -538.41, -2989.19, -1680.07, 
    703.46, -2502.55, -2018.70, -2262.50, -606.07, -3203.93, -2640.50, -1766.34, -3255.43, -601.20,
    -2249.83, -2579.85, -3729.54, -1175.59, -4238.81, -3227.92, -1839.42, -4556.34, -708.86, -653.48,
    -2116.89, -390.40, -2266.07, -1016.79, -364.49, -2034.71, -3824.05, -1285.70, -1224.47, -2959.95,
    -1492.64, -1065.22, -630.63, -1819.31, -156.23, -3083.18, -529.24, -2179.99, -2096.65, -3260.04,
    -284.38, -1720.55, -14.56, -1158.81, -2272.62
]

# 🔹 Średnia ruchoma (uśrednianie wyników na 5 epizodów)
window_size = 5
rolling_mean = np.convolve(total_rewards, np.ones(window_size)/window_size, mode='valid')

# 🔹 Rysowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(episodes, total_rewards, marker='o', linestyle='-', color='b', label="Całkowita nagroda")
plt.plot(episodes[window_size-1:], rolling_mean, linestyle='--', color='r', label="Średnia ruchoma (5 epizodów)")

# 🔹 Opisy osi
plt.xlabel("Epizod")
plt.ylabel("Całkowita nagroda")
plt.title("Postęp nauki modelu w kolejnych epizodach")
plt.legend()
plt.grid()

# 🔹 Pokazanie wykresu
plt.show()
