import matplotlib.pyplot as plt

# Dati aggiornati
seriale_data = {
    "2D": 0.001533,
    "2D2": 0.000013,
    "10D": 0.000905,
    "20D": 0.031411,
    "100D": 0.136542,
    "100D2": 1.842681
}

mpi_data = {
    "2D": 0.001483,
    "2D2": 0.000622,
    "10D": 0.002237,
    "20D": 0.017570,
    "100D": 0.028294,
    "100D2": 0.164148
}

omp_data = {
    "2D": 0.000821,
    "2D2": 0.000528,
    "10D": 0.000858,
    "20D": 0.001008,
    "100D": 0.001438,
    "100D2": 0.009094
}

mpi_omp_data = {
    "2D": 0.002298,
    "2D2": 0.000661,
    "10D": 0.002430,
    "20D": 0.014266,
    "100D": 0.010206,
    "100D2": 0.087996
}
cuda_data = {
    "2D": 0.000821,
    "2D2": 0.000528,
    "10D": 0.000858,
    "20D": 0.001008,
    "100D": 0.001438,
    "100D2": 0.009094
}

# Creazione delle etichette e dei valori
inputs = list(seriale_data.keys())
seriale_values = [seriale_data[input] for input in inputs]
mpi_values = [mpi_data[input] for input in inputs]
omp_values = [omp_data[input] for input in inputs]
mpi_omp_values = [mpi_omp_data[input] for input in inputs]
cuda_values = [cuda_data[input] for input in inputs]
# Creazione del grafico
plt.figure(figsize=(12, 8))
plt.plot(inputs, seriale_values, marker='o', label='Seriale')
plt.plot(inputs, mpi_values, marker='o', label='MPI')
plt.plot(inputs, omp_values, marker='o', label='OpenMP')
plt.plot(inputs, mpi_omp_values, marker='o', label='MPI + OpenMP')
plt.plot(inputs, cuda_values, marker='o', label='CUDA')
# Dettagli del grafico
plt.title("Confronto dei Tempi di Esecuzione: Seriale, MPI, OpenMP e MPI + OpenMP")
plt.xlabel("Input")
plt.ylabel("Tempo (s)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Grafico_Tempi_Confronto.png")
plt.show()
