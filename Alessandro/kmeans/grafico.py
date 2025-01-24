import matplotlib.pyplot as plt

# Definizione dei dati dal file Tempi.md
mpi_non_opt_data = {
    "2D": [0.001697, 0.003303],
    "2D2": [0.000062, 0.001132],
    "10D": [0.001745, 0.004330],
    "20D": [0.002923, 0.011930],
    "100D": [0.013726, 0.040489],
    "100D2": [0.229977, 0.375605]
}

mpi_opt_data = {
    "2D": [0.004095, 0.005206],
    "2D2": [0.000051, 0.001058],
    "10D": [0.001654, 0.006909],
    "20D": [0.002952, 0.011499],
    "100D": [0.009717, 0.029113],
    "100D2": [0.067982, 0.146152]
}

# Conversione dei dati per il grafico
inputs = list(mpi_non_opt_data.keys())
non_opt_values = [sum(values) / len(values) for values in mpi_non_opt_data.values()]
opt_values = [sum(values) / len(values) for values in mpi_opt_data.values()]

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.plot(inputs, non_opt_values, marker='o', label='MPI Non Ottimizzato')
plt.plot(inputs, opt_values, marker='o', label='MPI Ottimizzato')
plt.title("Confronto tra MPI Non Ottimizzato e Ottimizzato")
plt.xlabel("Input")
plt.ylabel("Tempo Medio (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Mostra il grafico
plt.show()
