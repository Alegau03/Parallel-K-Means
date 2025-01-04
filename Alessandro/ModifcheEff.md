# VERSIONE OPENMP
# 1. Parallelizzazione del calcolo delle distanze e dell'assegnazione dei cluster
Codice originale:

```C
for(int i = 0; i < lines; i++) {
    int class = -1;
    float minDist = FLT_MAX;
    for(int j = 0; j < K; j++) {
        float dist = euclideanDistance(&data[i * samples], &centroids[j * samples], samples);
        if(dist < minDist) {
            minDist = dist;
            class = j + 1;
        }
    }
    if(classMap[i] != class) {
        changes++;
    }
    classMap[i] = class;
}
```
## Parallelizzazione:
Ho utilizzato un ciclo `#pragma omp parallel` for con la direttiva `reduction(+:changes)` per calcolare i cambiamenti in parallelo.

```C
#pragma omp parallel for reduction(+:changes) schedule(static)
for(int i = 0; i < lines; i++) {
    int class = -1;
    float minDist = FLT_MAX;

    for(int j = 0; j < K; j++) {
        float dist = euclideanDistance(&data[i * samples], &centroids[j * samples], samples);
        if(dist < minDist) {
            minDist = dist;
            class = j + 1;
        }
    }
    if(classMap[i] != class) {
        changes++;
    }
    classMap[i] = class;
}
```
### Motivazione:

Ogni punto (lines) può essere processato indipendentemente per calcolare la distanza da tutti i centroidi.
La riduzione su changes garantisce che i risultati siano correttamente accumulati.

# 2. Parallelizzazione del ricalcolo dei centroidi
Codice originale:
```C
zeroIntArray(pointsPerClass, K);
zeroFloatMatrix(auxCentroids, K, samples);

for(int i = 0; i < lines; i++) {
    int class = classMap[i] - 1;
    pointsPerClass[class]++;
    for(int j = 0; j < samples; j++) {
        auxCentroids[class * samples + j] += data[i * samples + j];
    }
}

for(int i = 0; i < K; i++) {
    for(int j = 0; j < samples; j++) {
        auxCentroids[i * samples + j] /= pointsPerClass[i];
    }
}
```
## Parallelizzazione: 
Ho separato la somma delle coordinate per ogni cluster e il conteggio dei punti con direttive #pragma omp parallel for. Ho usato #pragma omp atomic per gestire correttamente gli aggiornamenti concorrenti.

```C
zeroIntArray(pointsPerClass, K);
zeroFloatMatrix(auxCentroids, K, samples);

#pragma omp parallel for schedule(static)
for(int i = 0; i < lines; i++) {
    int class = classMap[i] - 1;
    #pragma omp atomic
    pointsPerClass[class]++;

    for(int j = 0; j < samples; j++) {
        #pragma omp atomic
        auxCentroids[class * samples + j] += data[i * samples + j];
    }
}

#pragma omp parallel for schedule(static)
for(int i = 0; i < K; i++) {
    for(int j = 0; j < samples; j++) {
        if(pointsPerClass[i] > 0) {
            auxCentroids[i * samples + j] /= pointsPerClass[i];
        }
    }
}
```
### Motivazione:

La somma e il conteggio possono essere effettuati in parallelo, ma richiedono sincronizzazione (atomic) per evitare corruzione dei dati.
# 3. Parallelizzazione del calcolo della distanza massima tra centroidi
Codice originale:
```C
maxDist = FLT_MIN;
for(int i = 0; i < K; i++) {
    distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
    maxDist = MAX(maxDist, distCentroids[i]);
}
```
## Parallelizzazione:
Ho usato una direttiva #pragma omp parallel for con reduction(max:maxDist) per calcolare in parallelo la distanza massima.

```C
maxDist = FLT_MIN;
#pragma omp parallel for reduction(max:maxDist) schedule(static)
for(int i = 0; i < K; i++) {
    distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
    maxDist = MAX(maxDist, distCentroids[i]);
}
```
### Motivazione:

Ogni distanza può essere calcolata indipendentemente.
La riduzione su maxDist garantisce il corretto calcolo del massimo valore.
# 4. Parallelizzazione dell'aggiornamento dei centroidi
Codice originale:
```C
for(int i = 0; i < K * samples; i++) {
    centroids[i] = auxCentroids[i];
}
```
## Parallelizzazione:
```C
#pragma omp parallel for schedule(static)
for(int i = 0; i < K * samples; i++) {
    centroids[i] = auxCentroids[i];
}
```
### Motivazione:

Ogni aggiornamento è indipendente, quindi il ciclo è naturalmente parallelizzabile.


# Strategie usate per aumentare le performance
## Bilanciamento del carico:
Ho usato schedule(static) per distribuire uniformemente il lavoro tra i thread, evitando squilibri.

## Uso delle direttive OpenMP:
`reduction`: Per sommare changes e calcolare il massimo valore di maxDist.
`atomic`: Per evitare conflitti negli aggiornamenti di array condivisi (pointsPerClass e auxCentroids).

## Minimizzazione della sincronizzazione:
Ho ridotto al minimo l'uso di operazioni sincronizzate (es. atomic) limitandole a punti critici.
## Scalabilità:
Ho parallelizzato tutti i principali loop computazionali, consentendo al programma di scalare su più core.



# VERSIONE MPI
## Inizializzazione MPI
Viene chiamato MPI_Init per avviare l’ambiente MPI. Si recuperano rank (identificativo del processo) e numProcs (numero totale di processi MPI).

## Lettura dei parametri
Solo il rank 0 legge (da riga di comando e da file) le informazioni:

Nome file di input (contenente i punti).
K: numero di cluster.
maxIterations: massimo numero di iterazioni consentito.
minChanges: percentuale minima di cambiamenti di cluster che, se non superata, causa la terminazione.
maxThreshold: soglia massima di spostamento dei centroidi per fermare l'algoritmo.
Quindi, questi valori vengono broadcast a tutti i processi.

## Lettura e distribuzione dei dati

Il rank 0 legge l’intero dataset nel proprio array data.
Con MPI_Bcast, si replica l’intero array data in ogni processo (soluzione semplice, adatta a dataset non eccessivamente grandi).
## Inizializzazione dei centroidi

Il rank 0 sceglie in modo casuale K indici di punti come centroidi iniziali, e li copia in centroids.
Con MPI_Bcast, i centroidi iniziali sono inviati a tutti i processi.

## Suddivisione del carico

Si calcola startIndex e endIndex per ogni processo, in modo da suddividere i punti (le righe del dataset) tra i processi.
Ognuno gestisce i propri myNumPoints = endIndex - startIndex.
Main Loop k-means
Si ripete finché non si incontra la condizione di terminazione:

## Assegnamento dei punti
Ogni processo, per i propri punti, calcola la distanza da ciascun centroide e assegna il punto al cluster più vicino. Se un punto cambia cluster, incrementa un contatore di cambiamenti localChanges.

## Accumulo
Per ogni cluster, si somma il contributo delle coordinate dei punti, da usare poi per calcolare la media.
## Riduzioni MPI:
MPI_Allreduce per sommare tutti i localChanges in globalChanges.
MPI_Reduce per sommare i vettori pointsPerClassLocal (numero di punti per cluster) e auxCentroidsLocal (somma delle coordinate) in rank 0.

## Aggiornamento dei centroidi (solo in rank 0):
Si calcola la media (nuovo centroide = somma / numero di punti) per ogni cluster.
Si calcola la distanza tra vecchi e nuovi centroidi; si memorizza la distanza massima in maxDist.
Se maxDist <= maxThreshold oppure se globalChanges <= minChanges o se si è raggiunto maxIterations, si pone fine.
Il rank 0 trasmette (con MPI_Bcast) un flag continueFlag che indica se continuare o meno. Se si continua, vengono diffusi i centroidi aggiornati.
## Raccolta dei risultati

Con **MPI_Gatherv**, il rank 0 si fa inviare da ogni processo le assegnazioni finali (classMapLocal) per costruire un classMapGlobal coerente per tutti i punti.
Infine, solo in rank 0 viene scritto su file l’ID di cluster di ogni punto.
## Deallocazione e fine

Si liberano le risorse allocate, e si chiude l’ambiente MPI con MPI_Finalize().
In sintesi, la logica di k-means resta la stessa di un’implementazione sequenziale, ma qui è distribuita tra i processi MPI, che collaborano tramite collettive (MPI_Bcast, MPI_Reduce, MPI_Gatherv, ecc.) per sincronizzare i centroidi aggiornati e le assegnazioni dei punti.

In questo modo, ogni processo lavora su una porzione dei punti, riducendo il tempo totale di calcolo rispetto a un’esecuzione sequenziale

## Ottimizzazione
Funzione **distanceSquared** al posto di **euclideanDistance** durante l’assegnamento dei punti al cluster più vicino. Si confrontano i quadrati delle distanze senza calcolare la sqrtf, riducendo i *costi di calcolo*.
### Calcolo dello spostamento dei centroidi:
Manteniamo un buffer oldCentroids in rank 0. Prima di aggiornare i centroidi, copiamo i centroidi correnti in oldCentroids.
Calcoliamo poi la distanza reale (con sqrtf) solo dopo aver ricavato i nuovi centroidi, in modo da trovare il maxDist.
Questo maxDist viene confrontato con la maxThreshold.
Il resto (MPI_Reduce, calcolo delle medie, Gatherv, ecc.) rimane quasi invariato per mantenere la stessa struttura.
Queste modifiche, specialmente saltare la *sqrtf* durante l’assegnamento, permettono un notevole risparmio di cicli di CPU, aumentando la velocità dell’algoritmo. Inoltre, la copia degli oldCentroids e il successivo calcolo dello spostamento massimo riducono l’overhead di ricalcolare le distanze due volte.




# TEMPI ATTUALI
I tempi di MPi sono espressi con 4 processi 
## input2D.inp
| **Fase**               | **Versione Seriale**       | **Versione MPI**               | **Versione OpenMP**                |
|------------------------|----------------------------|--------------------------------|------------------------------------|
| **Initialization**     | 0.001719 s (1.0x)         | 0.002057 s (1.20x)             | 0.002698 s (1.57x)                 |
| **Computation**        | 0.002444 s (6.85x)        | 0.000357 s (1.0x)              | 0.005989 s (16.78x)                |
| **Somma (Init+Comp)**  | 0.004163 s (1.72x)        | 0.002414 s (1.0x)              | 0.008687 s (3.60x)                 |
| **Memory deallocation**| 0.000929 s                | N/A                            | N/A                                 |

## input100D.inp
| **Fase**                     | **Versione Seriale**         | **Versione MPI**           | **Versione OpenMP**         |
|-----------------------------:|:---------------------------:|:--------------------------:|:---------------------------:|
| **Memory allocation**        | 0.033667 s (1.31x)          | 0.025747 s (1.0x)          | 0.039986 s (1.55x)          |
| **Computation**              | 0.046523 s (2.72x)          | 0.017071 s (1.0x)          | 0.022172 s (1.30x)          |
| **Somma (Alloc+Comp)**       | 0.080193 s (1.87x)          | 0.042818 s (1.0x)          | 0.062158 s (1.45x)          |
| **Memory deallocation**      | 0.000678 s                  | N/A                        | N/A                         |