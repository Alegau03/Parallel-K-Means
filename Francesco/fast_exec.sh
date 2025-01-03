mpicc KMEANS_mpi.c -o kmeans_mpi 
mpicc kmeans_ALE.c -o kmeans_ale      
mpirun  -n 6   ./kmeans_mpi test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D  > result
#./kmeans_omp test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_omp

mpirun  -n 6   ./kmeans_ale test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_ale  > result_ale
gcc KMEANS.c -o kmeans
./kmeans test_files/input2D.inp 6 3000 10 0.00001 test_files/output2D_seq > result_seq
cd test_files
paste input2D.inp output2D > graph2D.txt   
paste input2D.inp output2D_ale > graph2D_ale.txt   
paste input2D.inp output2D_seq > graph2D_seq.txt 
#gnuplot -p plot_kmeans.gp 
#gnuplot -p plot_kmeans_seq.gp 
cd ..
output_file1="computation_performance"
output_file2="computation_performance_ale"

input_file1="result"
input_file2="result_ale"

grep "Computation" "$input_file1" | cut -d':' -f2  >> "$output_file1"
grep "Computation" "$input_file2" | cut -d':' -f2  >> "$output_file2"
