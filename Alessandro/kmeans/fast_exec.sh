mpicc -o kmeans_mpi KMEANS_mpi.c -lm 
echo "Inizio run versione MPI" 
mpirun -np 4 ./kmeans_mpi test_files/input2D.inp 6 300 10 0.0001 test_files/output2d_mpi.txt > result_mpi
echo "Fine run versione MPI"
gcc-14 -fopenmp -o kmeans_openmp KMEANS_omp.c -lm
echo "Inizio run versione OpenMP"
./kmeans_openmp test_files/input2D.inp 6 300 10 0.0001 test_files/output2d_omp.txt
echo "Fine run versione OpenMP"
gcc KMEANS.c -o kmeans
echo "Inizio run versione sequenziale"
./kmeans test_files/input2D.inp 6 300 10 0.0001 test_files/output2d_seq.txt
echo "Fine run versione sequenziale"
cd test_files
paste input2D.inp output2d_seq.txt > graph2d_seq.txt
paste input2D.inp output2d_mpi.txt > graph2d_mpi.txt
paste input2D.inp output2d_omp.txt > graph2d_omp.txt
gnuplot -p plot_kmeans_2d_seq.gp
gnuplot -p plot_kmeans_2d_mpi.gp
gnuplot -p plot_kmeans_2d_omp.gp
echo "Done"