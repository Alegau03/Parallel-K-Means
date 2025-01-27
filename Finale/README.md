# K-Means Clustering Parallel Implementation

Questo script compila e esegue diverse implementazioni parallele e sequenziali dell'algoritmo K-Means per il clustering. Supporta implementazioni in **MPI**, **OpenMP**, **MPI+OpenMP**, e una versione sequenziale.

## Requisiti

- **Compilatori**:
  - `mpicc` (MPI compiler)
  - `gcc` con supporto OpenMP
- **File di input/output**:
  - `test_files/input100D2.inp` (file di input con i dati da processare)
  - Output salvati in `test_files/` come file separati.
- **Librerie**:
  - MPI
  - OpenMP
  - Libreria matematica (`-lm`)

## Uso

1. Assicurati che i compilatori necessari e le librerie siano installati.
2. Rendi eseguibile lo script:
   ```bash
   chmod +x script.sh
   ```
3. Esegui lo script con:
    ```bash
   ./run.sh
   ```

## Funzionalità dello script
### Compilazione: Lo script compila i seguenti file:
- KMEANS_mpi.c in un eseguibile chiamato kmeans_mpi
- KMEANS_omp.c in un eseguibile chiamato kmeans_openmp
- KMEANS_mpi_omp.c in un eseguibile chiamato kmeans_mpi_omp
- KMEANS.c in un eseguibile chiamato kmeans

### Esecuzione: 
Lo script esegue ogni programma n volte per valutare i tempi di esecuzione e confrontare i risultati. 
I dettagli:
- Input: test_files/inputDesiderato
- Parametri (Modificabili a piacere):
  - Numero di cluster: 6
  - Iterazioni massime: 300
  - Cambiamenti minimi: 1
  - Soglia: 0.0001
- Output:
  - test_files/output_mpi.txt
  - test_files/output_omp.txt
  - test_files/output100d2_seq.txt
  - test_files/output_mpi_omp.txt
- Risultati:
  - Tempi di esecuzione salvati in:
    - test_files/timing_mpi.txt
    - test_files/timing_omp.txt
    - test_files/timing_seq.txt
    - test_files/timing_mpi_omp.txt


### Analisi dei risultati
Lo script include comandi per confrontare i file di output generati e preparare dati per visualizzazioni grafiche (Funzionante solo su input2D.inp).

## Visualizzazione: 
Al termine dell'esecuzione, i tempi vengono analizzati utilizzando un file Python (tempi.py) per la generazione di report.