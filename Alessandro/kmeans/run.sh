#!/bin/bash

# Compilazione dei programmi
echo "Compilazione dei programmi..."
mpicc -o kmeans_mpi KMEANS_mpi.c -lm 
gcc-14 -fopenmp -o kmeans_openmp KMEANS_omp.c -lm
gcc KMEANS.c -o kmeans

# Parametri comuni
INPUT_FILE="test_files/input2D.inp"
OUTPUT_FILE_MPI="test_files/output2d_mpi.txt"
OUTPUT_FILE_OMP="test_files/output2d_omp.txt"
OUTPUT_FILE_SEQ="test_files/output2d_seq.txt"
NUM_CLUSTER=6
MAX_ITERATIONS=300
MIN_CHANGES=10
THRESHOLD=0.0001

# File di output per i risultati e i tempi
RESULT_MPI="test_files/result_mpi"
RESULT_OMP="test_files/result_omp"
RESULT_SEQ="test_files/result_seq"
TIMING_MPI="test_files/timing_mpi.txt"
TIMING_OMP="test_files/timing_omp.txt"
TIMING_SEQ="test_files/timing_seq.txt"

# Inizializza i file dei tempi
> $TIMING_MPI
> $TIMING_OMP
> $TIMING_SEQ

# Funzione per eseguire un programma 1000 volte
run_program() {
    PROGRAM=$1      # Nome del programma da eseguire
    ARGS=$2         # Argomenti del programma
    OUTPUT=$3       # File di output del programma
    TIMING_FILE=$4  # File per salvare i tempi di computazione

    echo "Esecuzione di $PROGRAM per 1000 iterazioni..."
    for ((i=1; i<=100; i++)); do
        # Esegui il programma e salva l'output
        $PROGRAM $ARGS > "$OUTPUT"
        # Estrai il tempo di computazione e salvalo
        grep "Computation:" "$OUTPUT" | tail -n 1 | awk '{print $2}' >> "$TIMING_FILE"
    done
    echo "Esecuzione di $PROGRAM completata!"
}

# Esegui i programmi
run_program "mpirun -np 4 ./kmeans_mpi" "$INPUT_FILE $NUM_CLUSTER $MAX_ITERATIONS $MIN_CHANGES $THRESHOLD $OUTPUT_FILE_MPI" "$RESULT_MPI" "$TIMING_MPI"
run_program "./kmeans_openmp" "$INPUT_FILE $NUM_CLUSTER $MAX_ITERATIONS $MIN_CHANGES $THRESHOLD $OUTPUT_FILE_OMP" "$RESULT_OMP" "$TIMING_OMP"
run_program "./kmeans" "$INPUT_FILE $NUM_CLUSTER $MAX_ITERATIONS $MIN_CHANGES $THRESHOLD $OUTPUT_FILE_SEQ" "$RESULT_SEQ" "$TIMING_SEQ"

echo "Tutte le esecuzioni sono state completate!"
