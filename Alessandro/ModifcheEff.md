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