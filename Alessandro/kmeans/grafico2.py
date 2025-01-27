import matplotlib.pyplot as plt

# Dati forniti
seriale_data = {
   "2D":0.001022,
"10D":0.000345,
"20D":0.011799,
"100D":0.020280,
"100D2":0.256416,
}

mpi_data = {
    "2D": [0.001022,0.001493, 0.001487, 0.004095, 0.002332, 0.002867, 0.003209, 0.005206],
    "10D": [0.000345,0.001226, 0.001261, 0.001654, 0.001981, 0.002146, 0.002781, 0.006909],
    "20D": [0.011799,0.004959, 0.003669, 0.002952, 0.007657, 0.011217, 0.011975, 0.011499],
    "100D": [0.020280,0.015018, 0.010801, 0.009717, 0.018994, 0.018681, 0.019969, 0.029113],
    "100D2": [0.256416,0.110244, 0.080355, 0.067982, 0.123403, 0.122489, 0.119901, 0.146152]
}
num_processi = [1, 2, 3, 4, 5, 6, 7, 8]

# Calcolo dell'efficienza
# Efficienza è definita come (Tempo seriale / Tempo parallelo) / Numero di processi
efficiency_data = {}
for input_type, mpi_times in mpi_data.items():
    serial_time = seriale_data[input_type]
    efficiency = [((serial_time / mpi_time) ) for mpi_time in mpi_times]
    efficiency_data[input_type] = efficiency

# Creazione del grafico
plt.figure(figsize=(12, 8))
for input_type, efficiency in efficiency_data.items():
    plt.plot(num_processi, efficiency, marker='o', label=f"Speedup - {input_type}")

plt.title("Speedup del Programma MPI")
plt.xlabel("Numero di Processi")
plt.ylabel("Speedup")
plt.legend(title="Input")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Speedup.png")
plt.show()
