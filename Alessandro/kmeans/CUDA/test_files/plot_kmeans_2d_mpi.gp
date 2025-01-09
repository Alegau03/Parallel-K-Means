set title "KMeans Clusters MPI"
set xlabel "X-axis"
set ylabel "Y-axis"
set key off
set palette model RGB defined (0 "red", 1 "blue", 2 "green", 3 "orange", 4 "purple")

plot "graph2d_mpi.txt" using 1:2:3 with points pt 7 palette
