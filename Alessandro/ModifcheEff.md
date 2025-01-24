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
### INPUT 2D2
0.000002
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


CON OTTIMIZZAZIONE SULLA DISTANZA
### INPUT 2D
4 Processi: 0.004095
8 Processi: 0.005206
### INPUT 2D2
4 Processi: 0.000051
8 Processi: 0.001058
### INPUT 10D
4 Processi: 0.001654
8 Processi: 0.006909 
### INPUT 20D
4 Processi: 0.002952
8 Processi: 0.011499
### INPUT 100D
4 Processi: 0.009717
8 Processi: 0.029113
### INPUT 100D2
4 Processi: 0.067982
8 Processi: 0.146152

## Alcuni test su MPI
Faccio delle esecuzioni con altri parametri, saranno tutti svolti con MPI ottimizzato con 4 thread (il più veloce) su input 100D2

### Prima Esecuzione
- NUM_CLUSTER=100
- MAX_ITERATIONS=300
- MIN_CHANGES=1
- THRESHOLD=0.0001
Tempo: 2.461747

### Seconda Esecuzione
- NUM_CLUSTER=1000
- MAX_ITERATIONS=300
- MIN_CHANGES=1
- THRESHOLD=0.0001
Tempo: 5.804244

### Terza Esecuzione
- NUM_CLUSTER=1000
- MAX_ITERATIONS=300
- MIN_CHANGES=0.001
- THRESHOLD=0.0001
Tempo: 21.353719

### Quarta Esecuzione
- NUM_CLUSTER=10000
- MAX_ITERATIONS=300
- MIN_CHANGES=1
- THRESHOLD=0.0001
Tempo: 30.266113

### Quinta Esecuzione
- NUM_CLUSTER=10000
- MAX_ITERATIONS=300
- MIN_CHANGES=0.001
- THRESHOLD=0.0001
Tempo: 115.531722
## OMP

CON OTTIMIZZAZIONE SULLA DISTANZA

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

### INPUT 2D2

### INPUT 10D

### INPUT 20D

### INPUT 100D

### INPUT 100D2


## CUDA