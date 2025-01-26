import matplotlib.pyplot as plt

# Dati forniti corretti
mpi_omp_data = {
    "2D": 0.005996,
    "2D2": 0.000398,
    "10D": 0.003921,
    "20D": 0.009482,
    "100D": 0.013373,
    "100D2": 0.079466
}

omp_data = {
    "2D": 0.001015,
    "2D2": 0.000257,
    "10D": 0.000796,
    "20D": 0.004708,
    "100D": 0.008996,
    "100D2": 0.081544
}

mpi_4proc_data = {
    "2D": 0.004095,
    "2D2": 0.000051,
    "10D": 0.001654,
    "20D": 0.002952,
    "100D": 0.009717,
    "100D2": 0.067982
}

# Creazione delle etichette e dei valori
inputs = list(mpi_omp_data.keys())
mpi_omp_values = [mpi_omp_data[input] for input in inputs]
omp_values = [omp_data[input] for input in inputs]
mpi_values = [mpi_4proc_data[input] for input in inputs]

# Creazione del grafico
plt.figure(figsize=(12, 8))
plt.plot(inputs, mpi_omp_values, marker='o', label='MPI + OpenMP (4 processi)')
plt.plot(inputs, omp_values, marker='o', label='OpenMP')
plt.plot(inputs, mpi_values, marker='o', label='MPI (4 processi)')

# Dettagli del grafico
plt.title("Confronto dei Tempi di Esecuzione: MPI, OpenMP e MPI + OpenMP")
plt.xlabel("Input")
plt.ylabel("Tempo (s)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Grafico3.png")
plt.show()
