import matplotlib.pyplot as plt

# Dati forniti
seriale_data = {
    "2D": 0.001022,
    "10D": 0.000345,
    "20D": 0.011799,
    "100D": 0.020280,
    "100D2": 0.256416
}

mpi_data = {
    "2D": [0.001022, 0.005281, 0.004879, 0.003929, 0.002603, 0.001540, 0.001393, 0.004386],
    "10D": [0.000345, 0.004960, 0.003966, 0.002012, 0.002065, 0.001562, 0.002045, 0.003563],
    "20D": [0.011799, 0.013688, 0.008815, 0.007379, 0.008550, 0.006726, 0.007738, 0.010229],
    "100D": [0.020280, 0.016831, 0.012763, 0.010928, 0.017246, 0.015870, 0.016602, 0.021933],
    "100D2": [0.256416, 0.142089, 0.106194, 0.083641, 0.114023, 0.081003, 0.106479, 0.146113]
}
"""
"2D":0.001022
"10D":0.000345
"20D":0.011799
"100D":0.020280
"100D2":0.256416
"""
num_processi = [1,2, 3, 4, 5, 6, 7, 8]

# Calcolo dell'efficienza
# Efficienza è definita come (Tempo seriale / Tempo parallelo) / Numero di processi
efficiency_data = {}
for input_type, mpi_times in mpi_data.items():
    serial_time = seriale_data[input_type]
    efficiency = [((serial_time / mpi_time) /p ) for mpi_time,p in zip(mpi_times, num_processi)]
    efficiency_data[input_type] = efficiency

# Creazione del grafico
plt.figure(figsize=(12, 8))
for input_type, efficiency in efficiency_data.items():
    plt.plot(num_processi, efficiency, marker='o', label=f"Efficenza - {input_type}")

plt.title("Efficenza del Programma MPI")
plt.xlabel("Numero di Processi")
plt.ylabel("Efficenza")
plt.legend(title="Input")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Efficenza.png")
plt.show()
