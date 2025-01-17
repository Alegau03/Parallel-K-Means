def calcola_media(file_name):
    try:
        with open(file_name, 'r') as file:
            tempi = [float(line.strip()) for line in file.readlines() if line.strip()]
        
        numero_iterazioni = len(tempi)
        tempo_totale = sum(tempi)
        media_tempo = tempo_totale / numero_iterazioni if numero_iterazioni > 0 else 0
        
        print(f"Tempo per {numero_iterazioni} iterazioni di {file_name}: {media_tempo:.6f}")
    except FileNotFoundError:
        print(f"Errore: Il file {file_name} non è stato trovato.")
    except ValueError:
        print(f"Errore: Il file {file_name} contiene dati non validi.")
    except Exception as e:
        print(f"Errore inaspettato: {e}")

if __name__ == "__main__":
    # Nomi dei file
    file_seq = "test_files/timing_seq.txt"
    file_omp = "test_files/timing_omp.txt"
    file_mpi = "test_files/timing_mpi.txt"
    file_mpi_omp= "test_files/timing_mpi_omp.txt"
    # Calcola la media per ogni file
    for file_name in [file_seq, file_omp, file_mpi, file_mpi_omp]:
        calcola_media(file_name)
