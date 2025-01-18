#!/bin/bash

# Compilazione dei programmi
echo "##################################################"
echo "Compilazione dei programmi..."
mpicc -o kmeans_mpi KMEANS_mpi.c -lm 
gcc-14 -fopenmp -o kmeans_openmp KMEANS_omp.c -lm
mpicc -fopenmp -o kmeans_mpi_omp KMEANS_mpiOMP.c -lm
gcc KMEANS.c -o kmeans

# Parametri comuni
INPUT_FILE="test_files/input100D.inp"
OUTPUT_FILE_MPI="test_files/output2d_mpi.txt"
OUTPUT_FILE_OMP="test_files/output2d_omp.txt"
OUTPUT_FILE_SEQ="test_files/output2d_seq.txt"
OUTPUT_FILE_MPI_OMP="test_files/output2d_mpi_omp.txt"
NUM_CLUSTER=6
MAX_ITERATIONS=300
MIN_CHANGES=10
THRESHOLD=0.0001

# File di output per i risultati e i tempi
RESULT_MPI="test_files/result_mpi"
RESULT_OMP="test_files/result_omp"
RESULT_SEQ="test_files/result_seq"
RESULT_MPI_OMP="test_files/result_mpi_omp"
TIMING_MPI="test_files/timing_mpi.txt"
TIMING_OMP="test_files/timing_omp.txt"
TIMING_SEQ="test_files/timing_seq.txt"
TIMING_MPI_OMP="test_files/timing_mpi_omp.txt"

# Inizializza i file dei tempi
> $TIMING_MPI
> $TIMING_OMP
> $TIMING_SEQ
> $TIMING_MPI_OMP

# Funzione per eseguire un programma 1000 volte
run_program() {
    PROGRAM=$1      # Nome del programma da eseguire
    ARGS=$2         # Argomenti del programma
    OUTPUT=$3       # File di output del programma
    TIMING_FILE=$4  # File per salvare i tempi di computazione

    echo "Esecuzione di $PROGRAM per 100 iterazioni..."
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
run_program "mpirun -np 4 ./kmeans_mpi_omp" "$INPUT_FILE $NUM_CLUSTER $MAX_ITERATIONS $MIN_CHANGES $THRESHOLD $OUTPUT_FILE_MPI_OMP" "$RESULT_MPI_OMP" "$TIMING_MPI_OMP"
echo "##################################################"
echo "Tutte le esecuzioni sono state completate!"
cd test_files
echo "Differenze outputs"
echo "Differenze tra output sequenziale e MPI"
diff output2d_seq.txt output2d_mpi.txt
echo "Differenze tra output sequenziale e OpenMP"
diff output2d_seq.txt output2d_omp.txt
echo "Differenze tra output sequenziale e MPI+OpenMP"
diff output2d_seq.txt output2d_mpi_omp.txt
echo "Differenze tra output MPI e OpenMP"
diff output2d_mpi.txt output2d_omp.txt
echo "Differenze tra output MPI e MPI+OpenMP"
diff output2d_mpi.txt output2d_mpi_omp.txt
echo "Differenze tra output OpenMP e MPI+OpenMP"
diff output2d_omp.txt output2d_mpi_omp.txt
echo "##################################################"
echo "Visione gnuplot"

paste input2D.inp output2d_seq.txt > graph2d_seq.txt
paste input2D.inp output2d_mpi.txt > graph2d_mpi.txt
paste input2D.inp output2d_omp.txt > graph2d_omp.txt
paste input2D.inp output2d_mpi_omp.txt > graph2d_mpi_omp.txt
gnuplot -p plot_kmeans_2d_seq.gp
gnuplot -p plot_kmeans_2d_mpi.gp
gnuplot -p plot_kmeans_2d_omp.gp
gnuplot -p plot_kmeans_2d_mpi_omp.gp
echo "Fatto"
echo "##################################################"
echo "Tempi"
cd ..
python tempi.py
echo "##################################################"
