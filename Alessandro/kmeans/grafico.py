import matplotlib.pyplot as plt

# Dati forniti
processes = ["2 Processi", "3 Processi", "4 Processi", "5 Processi", "6 Processi", "7 Processi", "8 Processi"]
inputs = ["2D", "2D2", "10D", "20D", "100D", "100D2"]
data = {
    "2D": [0.001493, 0.001487, 0.004095, 0.002332, 0.002867, 0.003209, 0.005206],
    "2D2": [0.000037, 0.000041, 0.000051, 0.000223, 0.000238, 0.000446, 0.001058],
    "10D": [0.001226, 0.001261, 0.001654, 0.001981, 0.002146, 0.002781, 0.006909],
    "20D": [0.004959, 0.003669, 0.002952, 0.007657, 0.011217, 0.011975, 0.011499],
    "100D": [0.015018, 0.010801, 0.009717, 0.018994, 0.018681, 0.019969, 0.029113],
    "100D2": [0.110244, 0.080355, 0.067982, 0.123403, 0.122489, 0.119901, 0.146152]
}

# Creazione del grafico
plt.figure(figsize=(12, 8))
for input_label in inputs:
    plt.plot(processes, data[input_label], marker='o', label=f"Input {input_label}")

# Dettagli del grafico
plt.title("Tempi di Esecuzione per Numero di Processi (MPI Ottimizzato)")
plt.xlabel("Numero di Processi")
plt.ylabel("Tempo (s)")
plt.legend(title="Input", loc="upper left")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Grafico2.png")
plt.show()
