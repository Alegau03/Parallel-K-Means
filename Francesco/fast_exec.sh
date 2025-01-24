mpicc KMEANS_mpi.c -o kmeans_mpi -lm
gcc -O3 -fopenmp KMEANS_omp.c -o kmeans_omp -lm
mpirun  -n 8   ./kmeans_mpi test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D  > result_mpi
./kmeans_omp test_files/input100D2.inp 6000 3000 10 0.00001 test_files/output100D2_omp > result_omp

gcc -O3 KMEANS.c -o kmeans -lm
./kmeans test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_seq > result_seq
cd test_files
paste input2D.inp output2D_omp > graph2D.txt   
paste input2D.inp output2D_seq > graph2D_seq.txt 
gnuplot -p plot_kmeans.gp 
gnuplot -p plot_kmeans_seq.gp 
cd ..
output_file1="computation_performance"

input_file1="result_mpi"

grep "Computation" "$input_file1" | cut -d':' -f2  >> "$output_file1"


