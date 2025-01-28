# TEMPI
Tutte le volte 100 iterazioni, per il codice Seriale, MPI e OpenMP la compilazione è stata effettuata con il flag -O3 per garantire la massima ottimizzazione possibile
I parametri sono: 
- NUM_CLUSTER=6
- MAX_ITERATIONS=300
- MIN_CHANGES=1
- THRESHOLD=0.0001

## SERIALE
### INPUT 2D
0.000265
### INPUT 10D
0.000231
### INPUT 20D
0.006027
### INPUT 100D
0.044763
### INPUT 100D2
0.756609

## MPI
SENZA OTTIMIZZAZIONE SULLA DISTANZA
### INPUT 2D
4 Processi: 0.001697
8 Processi: 0.003303
### INPUT 2D2
4 Processi: 
8 Processi: 
### INPUT 10D
4 Processi: 0.001745
8 Processi: 0.004330
### INPUT 20D
4 Processi: 0.002923
8 Processi: 0.011930
### INPUT 100D
4 Processi: 0.013726
8 Processi: 0.040489
### INPUT 100D2
4 Processi: 0.229977
8 Processi: 0.375605

Con ottimizzazione
## MPI 
### INPUT 2D
2 Processi: 0.005281
3 Processi: 0.004879
4 Processi: 0.003929
5 Processi: 0.002603
6 Processi: 0.001540
7 Processi: 0.001393
8 Processi: 0.004386
### INPUT 10D
2 Processi: 0.004960
3 Processi: 0.003966
4 Processi: 0.002012
5 Processi: 0.002065
6 Processi: 0.001562
7 Processi: 0.002045
8 Processi: 0.003563
### INPUT 20D
2 Processi: 0.013688
3 Processi: 0.008815
4 Processi: 0.007379
5 Processi: 0.008550
6 Processi: 0.006726
7 Processi: 0.007738
8 Processi: 0.010229
### INPUT 100D
2 Processi: 0.016831
3 Processi: 0.012763
4 Processi: 0.010928
5 Processi: 0.017246
6 Processi: 0.015870
7 Processi: 0.016602
8 Processi: 0.021933
### INPUT 100D2
2 Processi: 0.142089
3 Processi: 0.106194
4 Processi: 0.083641
5 Processi: 0.114023
6 Processi: 0.081003
7 Processi: 0.106479
8 Processi: 0.146113

## OMP
### INPUT 2D
0.001015
### INPUT 2D2
0.000257
### INPUT 10D
0.000796
### INPUT 20D
0.004708
### INPUT 100D
0.008996
### INPUT 100D2
0.081544 

## MPI + OPENMP
### INPUT 2D
4 Processi: 0.010145
8 Processi: 0.009604
### INPUT 2D2
4 Processi: 0.000381
8 Processi: 0.001572
### INPUT 10D
4 Processi: 0.003353
8 Processi: 0.010671
### INPUT 20D
4 Processi: 0.009472
8 Processi: 0.023515
### INPUT 100D
4 Processi: 0.043497
8 Processi: 0.031229
### INPUT 100D2
4 Processi: 0.068657
8 Processi: 0.114677

## CUDA
### INPUT 2D

### INPUT 2D2

### INPUT 10D

### INPUT 20D

### INPUT 100D

### INPUT 100D2
