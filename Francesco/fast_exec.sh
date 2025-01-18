mpicc KMEANS_mpi.c -o kmeans_mpi 
mpirun  -n 8   ./kmeans_mpi test_files/input100D2.inp 6 3000 10 0.00001 test_files/output100D2  > result_mpi
#./kmeans_omp test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_omp > result_mpi

gcc KMEANS.c -o kmeans
./kmeans test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_seq > result_seq
cd test_files
paste input2D.inp output2D > graph2D.txt   
paste input2D.inp output2D_seq > graph2D_seq.txt 
#gnuplot -p plot_kmeans.gp 
#gnuplot -p plot_kmeans_seq.gp 
cd ..
output_file1="computation_performance"

input_file1="result_mpi"

grep "Computation" "$input_file1" | cut -d':' -f2  >> "$output_file1"


