import matplotlib.pyplot as plt

# Dati forniti
inputs = ["2D", "2D2", "10D", "20D", "100D", "100D2"]

# 10% changes
l1_enabled_10 = [0.000773, 0.000615, 0.000773, 0.001580, 0.002667, 0.018356]
l1_disabled_10 = [0.000820, 0.000619, 0.000782, 0.001917, 0.002878, 0.021541]

# 1% changes
l1_enabled_1 = [0.001089, 0.000660, 0.001102, 0.005188, 0.011247, 0.110712]
l1_disabled_1 = [0.001099, 0.000755, 0.001277, 0.005546, 0.012780, 0.131142]

# Creazione del grafico
plt.figure(figsize=(12, 8))

# Grafico per 10% changes
plt.plot(inputs, l1_enabled_10, marker='o', label='10% changes - L1 enabled')
plt.plot(inputs, l1_disabled_10, marker='o', label='10% changes - L1 disabled')

# Grafico per 1% changes
plt.plot(inputs, l1_enabled_1, marker='o', label='1% changes - L1 enabled')
plt.plot(inputs, l1_disabled_1, marker='o', label='1% changes - L1 disabled')

# Dettagli del grafico
plt.title("Confronto delle Prestazioni CUDA: L1 abilitata vs disabilitata")
plt.xlabel("Input")
plt.ylabel("Tempo (s)")
plt.legend(title="Configurazione")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Prestazioni_CUDA_L1.png")
plt.show()