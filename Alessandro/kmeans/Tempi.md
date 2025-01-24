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
4 Processi: 0.000062
8 Processi: 0.001132
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
4 Processi: 0.000059
8 Processi: 0.000899
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
4 Processi: 0.152670
8 Processi: 0.278053