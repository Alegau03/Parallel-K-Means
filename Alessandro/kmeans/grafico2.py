import matplotlib.pyplot as plt

# Dati forniti
mpi_non_opt_data = {
    "2D": [0.001697, 0.003303],
    "10D": [0.001745, 0.004330],
    "20D": [0.002923, 0.011930],
    "100D": [0.013726, 0.040489],
    "100D2": [0.229977, 0.375605]
}

mpi_opt_data = {
    "2D": [0.003929, 0.004386],
    "10D": [0.002012, 0.003563],
    "20D": [0.007379, 0.010229],
    "100D": [0.010928, 0.021933],
    "100D2": [0.083641, 0.146113]
}

inputs = ["2D", "10D", "20D", "100D", "100D2"]

# Creazione del grafico
plt.figure(figsize=(12, 8))

# Aggiunta delle linee per MPI non ottimizzato (4 e 8 processi)
non_opt_4 = [times[0] for times in mpi_non_opt_data.values()]
non_opt_8 = [times[1] for times in mpi_non_opt_data.values()]
plt.plot(inputs, non_opt_4, marker='o', label="MPI Non Ottimizzato - 4 Processi")
plt.plot(inputs, non_opt_8, marker='o', label="MPI Non Ottimizzato - 8 Processi")

# Aggiunta delle linee per MPI ottimizzato (4 e 8 processi)
opt_4 = [times[0] for times in mpi_opt_data.values()]
opt_8 = [times[1] for times in mpi_opt_data.values()]
plt.plot(inputs, opt_4, marker='o', label="MPI Ottimizzato - 4 Processi")
plt.plot(inputs, opt_8, marker='o', label="MPI Ottimizzato - 8 Processi")

# Dettagli del grafico
plt.title("Confronto Tempi di Esecuzione: MPI Non Ottimizzato vs Ottimizzato")
plt.xlabel("Input")
plt.ylabel("Tempo di Esecuzione (s)")
plt.legend(title="Configurazione")
plt.grid(True)
plt.tight_layout()

# Salvataggio del grafico
plt.savefig("images/Grafico1.png")
plt.show()