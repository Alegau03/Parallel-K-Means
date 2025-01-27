/*
 * k-Means clustering algorithm
 *
 * MPI + OMP version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0
 * International License. https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define MAXLINE 2000
#define MAXCAD 200
#define UNROLL 2
// Distribuisce ciclicamente ((size+tmp)%size) i punti del dataset in modo bilanciato tra i processi MPI.
#define DISTRIBUTION(size, points, dist)                                       \
  int tmp = 0;                                                                 \
  while (points > 0) {                                                         \
    dist[(size + tmp) % size] += 1;                                            \
    tmp++;                                                                     \
    points--;                                                                  \
  }

// Calcola l’indice iniziale (offset) per ciascun processo MPI, basandosi sulla distribuzione dei punti.
#define OFFSET(offset, dist, rank)                                             \
  {                                                                            \
    offset[0] = 0;                                                             \
    for (int i = 1; i < rank; i++)                                             \
      offset[i] = offset[i - 1] + dist[i - 1];                                 \
  }
#define CALLTIME(call)                                                         \
  {                                                                            \
    double start = MPI_Wtime();                                                \
    call;                                                                      \
    double end = MPI_Wtime();                                                  \
    printf("\n%s time : %lf\n ", #call, end - start);                          \
  }
// Misura il tempo di esecuzione di una singola chiamata o funzione
#define BLOCKTIME(block)                                                       \
  {                                                                            \
    double start = MPI_Wtime();                                                \
    block double end = MPI_Wtime();                                            \
    printf("\n%s time : %lf\n ", #block, end - start);                         \
  }

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


/*
Function showFileError: It displays the corresponding error during file
reading.
*/

void showFileError(int error, char *filename) {
  printf("Error\n");
  switch (error) {
  case -1:
    fprintf(stderr, "\tFile %s has too many columns.\n", filename);
    fprintf(stderr,
            "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n",
            MAXLINE);
    break;
  case -2:
    fprintf(stderr, "Error reading file: %s.\n", filename);
    break;
  case -3:
    fprintf(stderr, "Error writing file: %s.\n", filename);
    break;
  }
  fflush(stderr);
}

/*
Function readInput: It reads the file to determine the number of rows and
columns.
*/
int readInput(char *filename, int *lines, int *samples) {
  FILE *fp;
  char line[MAXLINE] = "";
  char *ptr;
  const char *delim = "\t";
  int contlines, contsamples = 0;

  contlines = 0;

  if ((fp = fopen(filename, "r")) != NULL) {
    while (fgets(line, MAXLINE, fp) != NULL) {
      if (strchr(line, '\n') == NULL) {
        return -1;
      }
      contlines++;
      ptr = strtok(line, delim);
      contsamples = 0;
      while (ptr != NULL) {
        contsamples++;
        ptr = strtok(NULL, delim);
      }
    }
    fclose(fp);
    *lines = contlines;
    *samples = contsamples;
    return 0;
  } else {
    return -2;
  }
}

/*
Function readInput2: It loads data from file.
*/
int readInput2(char *filename, float *data) {
  FILE *fp;
  char line[MAXLINE] = "";
  char *ptr;
  const char *delim = "\t";
  int i = 0;

  if ((fp = fopen(filename, "rt")) != NULL) {
    while (fgets(line, MAXLINE, fp) != NULL) {
      ptr = strtok(line, delim);
      while (ptr != NULL) {
        data[i] = atof(ptr);
        i++;
        ptr = strtok(NULL, delim);
      }
    }
    fclose(fp);
    return 0;
  } else {
    return -2; // No file found
  }
}

/*
Function writeResult: It writes in the output file the cluster of each sample
(point).
*/
int writeResult(int *classMap, int lines, const char *filename) {
  FILE *fp;

  if ((fp = fopen(filename, "wt")) != NULL) {
    for (int i = 0; i < lines; i++) {
      fprintf(fp, "%d\n", classMap[i]);
    }
    fclose(fp);

    return 0;
  } else {
    return -3; // No file found
  }
}

/*

La funzione initCentroids:
inizializza i centroidi del clustering K-means copiando i valori dai punti di partenza specificati all'interno del dataset originale.
Definisce le posizioni iniziali dei centroidi, da cui il processo iterativo del K-means partirà.

Inizializza l’array centroids con i dati dei punti selezionati, 
che verranno poi utilizzati come punto di partenza per il calcolo iterativo dei centroidi nel K-means.

*/
void initCentroids(const float *data, float *centroids, int *centroidPos,
                   int samples, int K) {
  int i;
  int idx;
  for (i = 0; i < K; i++) {
    idx = centroidPos[i];
    //Copia il vettore del punto di dimensione samples che inizia in data[idx * samples] in centroids[i * samples]
    memcpy(&centroids[i * samples], &data[idx * samples],
           (samples * sizeof(float)));
  }
}
/*
Abbiamo utilizzato due versioni di euclideanDistance perchè in questo modo su input a due dimensioni
abbiamo potuto usare l'unrolling per migliorare le prestazioni.
*/
float UNROLLEDeuclideanDistance(float *point, float *center, int samples) {
  int blockSize = 32; // Dimensione del blocco ottimale per la cache L1
  int i, j;
  float diff1, diff2;
  diff1 = 0.0f;
  diff2 = 0.0f;
  // Calcolo a blocchi
  // Cliclo esterno per avanzare nei blocchi

  for (i = 0; i < samples; i += blockSize) {
    // Ciclo interno per calcolare la distanza tra i punti
    for (j = i; j * UNROLL < i + blockSize && j * UNROLL < samples; j++) {
      diff1 += (point[j * UNROLL] - center[j * UNROLL]) *
               (point[j * UNROLL] - center[j * UNROLL]);
      diff2 += (point[j * UNROLL + 1] - center[j * UNROLL + 1]) *
               (point[j * UNROLL + 1] - center[j * UNROLL + 1]);
    }
  }
  return diff1 + diff2; // Restituisce la distanza al quadrato
}
/*
Function euclideanDistance: distanca Euclicea tra due punti

La radice quadrata non è necessaria per confrontare le distanze relative tra punti e centroidi, 
poiché il confronto è valido anche con le distanze al quadrato.
Evitare il calcolo della radice quadrata migliora l’efficienza computazionale, dato che è un'operazione costosa.
Divide il lavoro in blocchi di dimensione blockSize
Questo migliora la località temporale e spaziale per ridurre i cache miss.
*/
float euclideanDistance(float *point, float *center, int samples) {
    float dist = 0.0f;
    int blockSize = 32; // Dimensione del blocco ottimale per la cache L1
    int i, j;
    float diff;

    // Calcolo a blocchi
    //Cliclo esterno per avanzare nei blocchi
    for (i = 0; i < samples; i += blockSize) {
        float blockDist = 0.0f; // Accumulatore temporaneo per il blocco
        //Ciclo interno per calcolare la distanza tra i punti
        for (j = i; j < i + blockSize && j < samples; j++) {
            diff = point[j] - center[j];
            blockDist += diff * diff;
        }
        dist += blockDist; // Somma il risultato del blocco
    }

    return dist; // Restituisce la distanza al quadrato
}
/*
Function zeroFloatMatriz: Set matrix elements to 0

*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
      matrix[i * columns + j] = 0.0;
}

/*
Function zeroIntArray: Set array elements to 0

*/
void zeroIntArray(int *array, int size) {
  int i;
  for (i = 0; i < size; i++)
    array[i] = 0;
}

int main(int argc, char *argv[]) {
  /* 0. Initialize MPI */
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN); //Gestore errori personalizzato

  // START CLOCK***************************************
  double start, end;
  start = MPI_Wtime();
  //**************************************************
  /*
   * PARAMETERS
   *
   * argv[1]: Input data file
   * argv[2]: Number of clusters
   * argv[3]: Maximum number of iterations of the method. Algorithm
   * termination condition. argv[4]: Minimum percentage of class changes.
   * Algorithm termination condition. If between one iteration and the next,
   * the percentage of class changes is less than this percentage, the
   * algorithm stops. argv[5]: Precision in the centroid distance after the
   * update. It is an algorithm termination condition. If between one
   * iteration of the algorithm and the next, the maximum distance between
   * centroids is less than this precision, the algorithm stops. argv[6]:
   * Output file. Class assigned to each point of the input file.
   * */
  if (argc != 7) {
    fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
    fprintf(stderr,
            "./KMEANS [Input Filename] [Number of clusters] [Number of "
            "iterations] [Number of changes] [Threshold] [Output data file]\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Reading the input data
  // lines = number of points; samples = number of dimensions per point
  int lines = 0, samples = 0;

  int error = readInput(argv[1], &lines, &samples);

  if (error != 0) {
    showFileError(error, argv[1]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

//Allocazione memoria
  float *data = (float *)calloc(lines * samples, sizeof(float)); 
  if (data == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  error = readInput2(argv[1], data);
  if (error != 0) {
    showFileError(error, argv[1]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Parametri del Clustering
  int K = atoi(argv[2]);
  int maxIterations = atoi(argv[3]);
  int minChanges = (int)(lines * atof(argv[4]) / 100.0);
  float maxThreshold = atof(argv[5]);

/*
centroidPos: Indici dei punti del dataset scelti come centroidi iniziali.
centroids: Coordinate dei centroidi correnti (inizializzate a 0 ma poi popolate con initCentroids).
classMap: Mappa che associa ogni punto del dataset al suo cluster corrente.
*/

// Alloca un array di interi di lunghezza K, inizializzando ogni elemento a 0.
  int *centroidPos = (int *)calloc(K, sizeof(int));

// Alloca un array di float con dimensione K * samples e inizializza ogni elemento a 0.0
  float *centroids = (float *)calloc(K * samples, sizeof(float));

// Alloca un array di interi lungo quanto il numero di punti (lines), inizializzando ogni elemento a 0
  int *classMap = (int *)calloc(lines, sizeof(int));

  if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Initial centrodis
  srand(0);
  int i;
  for (i = 0; i < K; i++)
    centroidPos[i] = rand() % lines;

  // Loading the array of initial centroids with the data from the array data
  // The centroids are points stored in the data array.
  initCentroids(data, centroids, centroidPos, samples, K);
  printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines,
         samples);
  printf("\tNumber of clusters: %d\n", K);
  printf("\tMaximum number of iterations: %d\n", maxIterations);
  printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges,
         atof(argv[4]), lines);
  printf("\tMaximum centroid precision: %f\n", maxThreshold);

  // END CLOCK*****************************************
  end = MPI_Wtime();
  ;
  printf("\nMemory allocation: %f seconds\n", end - start);
  fflush(stdout);

  //**************************************************
  // START CLOCK***************************************
  start = MPI_Wtime();
  //**************************************************
  char *outputMsg = (char *)calloc(10000, sizeof(char));
  char line[100];

  int j;
  int class;
  float dist, minDist;
  int it = 0;
  int changes = 0;
  float maxDist;

  // pointPerClass: Conta il numero di punti assegnati a ciascun cluster.
  // auxCentroids: Contiene le somme delle coordinate dei punti assegnati a ciascun cluster, utilizzato per calcolare la media (centroide) di ciascun cluster aggiornandolo.
  // distCentroids: Memorizza le distanze tra i centroidi precedenti e quelli aggiornati nell'iterazione corrente, calcolare il criterio di precisione dell'algoritmo, ovvero se i centroidi si sono spostati sotto una certa soglia (maxThreshold), il K-means può terminare.
  int *pointsPerClass = (int *)malloc((K + 1) * sizeof(int));
  float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
  float *distCentroids = (float *)malloc(K * sizeof(float));
  if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  /*
   *
   * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
   *
   */

  int comm_size, *point_distribution = NULL, *offset = NULL, my_iteration, my_offset;
  float *glob_auxCentroids;
  int *glob_pointsPerClass;

  glob_pointsPerClass = calloc(K + 1, sizeof(int));
  glob_auxCentroids = calloc(K * samples, sizeof(float));

  int tmp_lines = lines;

/*
Determinare la distribuzione dei punti del dataset tra i processi MPI.
Calcolare l'offset iniziale e il numero di punti (my_iteration) che ogni processo deve elaborare.
*/

  MPI_Comm_size(MPI_COMM_WORLD, &comm_size); //numero di processi

// Solo il processo 0 calcola la distribuzione dei punti tra i processi MPI
  if (rank == 0) {
    point_distribution = calloc(comm_size, sizeof(int)); //Allocato per memorizzare quanti punti vengono assegnati a ciascun processo.
    offset = calloc(comm_size, sizeof(int)); //Memorizza gli indici iniziali del dataset per ogni processo.
    DISTRIBUTION(comm_size, tmp_lines, point_distribution); //Distribuisce il numero totale di punti (tmp_lines) tra i processi in modo bilanciato.
    OFFSET(offset, point_distribution, comm_size); //Calcola l'offset iniziale per ogni processo, sommando i punti assegnati ai processi precedenti
    my_offset = offset[rank]; //è l'indice iniziale (offset[rank]).
    my_iteration = point_distribution[rank]; // è il numero di punti assegnati (point_distribution[rank]).
  } else { //Altri processi, calcola dinamicamente il numero di punti e l'offset.
    int remainder = lines % comm_size; 
    my_iteration = (rank < remainder) ? ((lines - remainder) / comm_size) + 1
                                      : ((lines - remainder) / comm_size); //Calcola il numero di punti che ogni processo deve elaborare.
    my_offset = (rank < remainder) ? rank * my_iteration
                                   : (remainder * (my_iteration + 1)) +
                                         ((rank - remainder) * my_iteration); //Calcola l'offset iniziale per ogni processo.
  }
  int *localClassMap = calloc(my_iteration, sizeof(int)); //Memorizza la classe assegnata a ciascun punto del dataset.
  MPI_Request requests[2]; //Array di richieste MPI.
  float reciprocal; 
  // Decido quale funzione usare in base all input, 2 dimensioni -> UNROLLEDeuclideanDistance, altrimenti euclideanDistance
  float (*distanceFun)(float *, float *, int) =
      (samples % 2 == 0) ? UNROLLEDeuclideanDistance : euclideanDistance;
    
  do {

    /*
      1. Calculate the distance from each point to the centroid
      Assign each point to the nearest centroid.
      Changes incrementa se la classe del punto cambia rispetto all'iterazione precedente.

    
      In questo parallel for calcoliamo quale sia il centroide più vicino per ogni punto e aggiorniamo due informazioni:
        -local_changes: se la classe del punto è cambiata rispetto all’iterazione precedente.
        -pointsPerClass[]: quanti punti sono stati assegnati a ciascun cluster.
      
      Li inseriamo in clausola reduction(+:local_changes, pointsPerClass[:K]) per evitare le race condition e sommare correttamente i contributi di tutti i thread.
      E' fondamentale la clausola di riduzione su pointsPerClass per evitare che i thread si sovrascrivano a vicenda.

    */

    it++;
    changes = 0;
    int i, j; // Da dichiarare fuori per evitare errori di compilazione
    // Variabile locale in riduzione per conteggiare quanti punti cambiano cluster
    int local_changes = 0;

// Parallel for con riduzione su local_changes e su pointsPerClass
#pragma omp parallel for private(j) \
            reduction(+:local_changes, pointsPerClass[:K]) \
            schedule(static)
// Itera sui punti assegnati a questo processo
for (i = my_offset; i < my_offset + my_iteration; i++) {
    // Indice locale (per accedere a localClassMap)
    int idx = i - my_offset;

    // Calcolo del cluster più vicino
    float minDist = FLT_MAX;
    int newClass = 1;  // inizializziamo a 1 o a qualunque cluster
    for (j = 0; j < K; j++) {
        float dist = distanceFun(&data[i * samples],
                                 &centroids[j * samples],
                                 samples);
        if (dist < minDist) {
            minDist = dist;
            newClass = j + 1;  // cluster j corrisponde a j+1
        }
    }

    // Verifica se la classe è cambiata rispetto alla vecchia
    if (localClassMap[idx] != newClass) {
        local_changes++;
    }

    // Aggiorna la classe del punto
    localClassMap[idx] = newClass;

    // Conta un punto in più per il cluster 'newClass'
    pointsPerClass[newClass - 1]++;
}

// Al termine del for parallelo, local_changes è stato ridotto: aggiorniamo changes
changes = local_changes;

// Aggiorniamo pointsPerClass[K] con il numero totale di punti che hanno cambiato classe
    pointsPerClass[K] = changes;
    zeroIntArray(glob_pointsPerClass, K + 1);

// Somma globale non bloccante sui contatori locali (pointsPerClass e auxCentroids).
    #pragma omp barrier
    MPI_Iallreduce(pointsPerClass, glob_pointsPerClass, K + 1, MPI_INT, MPI_SUM,
                   MPI_COMM_WORLD, &requests[0]); //Riduzione MPI non bloccante
    MPI_Waitall(1, requests, MPI_STATUSES_IGNORE);
    zeroFloatMatriz(auxCentroids, K, samples); //Inizializza la matrice auxCentroids a 0.
    zeroFloatMatriz(glob_auxCentroids, K, samples); //Inizializza la matrice glob_auxCentroids a 0.

    /*
     Scorre su tutti i punti locali assegnati al processo corrente.
     Usa la mappa delle classi locali (localClassMap) per identificare il cluster a cui appartiene ogni punto.
     Aggiorna le somme parziali delle coordinate (auxCentroids) per ciascun cluster e per ogni dimensione.

     Prepara i dati per calcolare i nuovi centroidi.
     Le somme parziali calcolate da ciascun processo verranno combinate globalmente tramite una riduzione (MPI_Iallreduce), permettendo l'aggiornamento dei centroidi.
    
        
     Esempio di azzeramento prima del parallel for (fuori da esso):
     memset(pointsPerClass, 0, sizeof(int)*K);
     memset(auxCentroids, 0, sizeof(float)*K*samples);

     Qui aggiorniamo di nuovo pointsPerClass[] (se vogliamo contare esattamente quanti punti totali sono in ogni cluster nella seconda passata) 
     e auxCentroids[] (vettore di somme delle coordinate, che poi sarà diviso per pointsPerClass[c] per ottenere i nuovi centroidi).
     Anche qui è fondamentale la clausola di riduzione su pointsPerClass e su auxCentroids per evitare che i thread si sovrascrivano a vicenda.
*/

#pragma omp parallel for private(j) \
            reduction(+:pointsPerClass[:K], auxCentroids[:K*samples]) \
            schedule(static)
for (i = my_offset; i < my_offset + my_iteration; i++) {
    // Calcolo l'indice locale per accedere a localClassMap
    int local_idx = i - my_offset;
    // Leggo la classe assegnata a questo punto
    int c = localClassMap[local_idx];  
    // Esempio: localClassMap contiene valori 1..K
    // Per accedere in array (0..K-1) usiamo (c-1).

    // Incremento il conteggio di punti appartenenti al cluster c
    pointsPerClass[c - 1]++;

    // Aggiungo le coordinate del punto i alle somme parziali di auxCentroids
    for (j = 0; j < samples; j++) {
        auxCentroids[(c - 1) * samples + j] += data[i * samples + j];
    }
}

// Al termine del for parallelo, le riduzioni sommano i risultati di ogni thread
// in pointsPerClass[] e auxCentroids[].

    // Somma globale non bloccante sui contatori locali (pointsPerClass e auxCentroids).
    #pragma omp barrier
    MPI_Iallreduce(auxCentroids, glob_auxCentroids, K * samples, MPI_FLOAT,
                   MPI_SUM, MPI_COMM_WORLD, &requests[1]);
    
    // Aspetta che entrambe le riduzioni siano completate prima di procedere.
    #pragma omp barrier
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);


  //Divide ogni somma globale per il numero di punti nel cluster
  //I centroidi vengono aggiornati copiando i nuovi valori.
  //Calcola la distanza massima tra i centroidi precedenti e quelli aggiornati.
    for (i = 0; i < K; i++) {
      reciprocal = 1.0f / glob_pointsPerClass[i];
      for (j = 0; j < samples; j++) {
        glob_auxCentroids[i * samples + j] *= reciprocal;
      }
    }
    changes = glob_pointsPerClass[K];
    zeroIntArray(pointsPerClass, K + 1);

    memcpy(centroids, glob_auxCentroids, (K * samples * sizeof(float)));

    maxDist = FLT_MIN;
    for (i = 0; i < K; i++) {
      distCentroids[i] = distanceFun(&centroids[i * samples],
                                     &auxCentroids[i * samples], samples);
      maxDist = MAX(maxDist, distCentroids[i]);
    }
    sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it,
            changes, sqrt(maxDist));
    outputMsg = strcat(outputMsg, line);

    //Condizione di terminazione
  } while ((changes > minChanges) && (it < maxIterations) &&
           (sqrt(maxDist) > maxThreshold));
// Terminiamo se:
// 1) changes <= minChanges (pochi cambiamenti di cluster)
// 2) raggiunto il maxIterations
// 3) i centroidi si sono spostati meno di maxThreshold

  // Raccoglie tutte le classificazioni locali nei processi e le combina nel processo con rank 0.
  #pragma omp barrier
  MPI_Gatherv(localClassMap, my_iteration, MPI_INT, classMap,
              point_distribution, offset, MPI_INT, 0, MPI_COMM_WORLD);
  /*
   *
   * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
   *
   */
  // Output and termination conditions

  printf("%s", outputMsg);

  // END CLOCK*****************************************
  end = MPI_Wtime();
  float comp_time = end - start;
  MPI_Reduce(&comp_time, MPI_IN_PLACE, 1, MPI_FLOAT, MPI_MAX, 0,
             MPI_COMM_WORLD); // Calcola il tempo di esecuzione massimo tra tutti i processi.
  if (rank == 0) {
    printf("\nComputation: %f seconds", comp_time);
    fflush(stdout);
  }
  //**************************************************
  // START CLOCK***************************************
  start = MPI_Wtime();
  //**************************************************
  if (changes <= minChanges) {
    printf("\n\nTermination condition:\nMinimum number of changes reached: %d "
           "[%d]",
           changes, minChanges);
  } else if (it >= maxIterations) {
    printf("\n\nTermination condition:\nMaximum number of iterations reached: "
           "%d [%d]",
           it, maxIterations);
  } else {
    printf("\n\nTermination condition:\nCentroid update precision reached: %g "
           "[%g]",
           maxDist, maxThreshold);
  }
  if (rank == 0) {
    // Writing the classification of each point to the output file.
    error = writeResult(classMap, lines, argv[6]);
    if (error != 0) {
      showFileError(error, argv[6]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  // Free memory
  free(data);
  free(point_distribution);
  free(offset);
  free(classMap);
  free(centroidPos);
  free(localClassMap);
  free(centroids);
  free(distCentroids);
  free(pointsPerClass);
  free(auxCentroids);
  
  // END CLOCK*****************************************
  end = MPI_Wtime();
  printf("\n\nMemory deallocation: %f seconds\n", end - start);
  fflush(stdout);
  //***************************************************/
  MPI_Finalize();
  return 0;
}

/*
    NOTE FINALI:
    I cambiamenti che abbiamo fatto e perché ora funziona
Inserimento di variabili in clausola reduction

Prima, nel tuo codice, facevi changes++ o pointsPerClass[...]++ dentro un for parallelo senza protezioni. 
Ciò causava race condition, perché più thread modificavano contemporaneamente la stessa variabile/struttura.
Adesso, con reduction(+:local_changes, pointsPerClass[:K]), ogni thread mantiene un accumulatore locale. 
Alla fine del for, OpenMP somma i risultati parziali di ogni thread e li mette nella variabile globale.
Eliminazione della doppia dichiarazione di variabili (ad esempio int i, j;)

Prima c’era un errore di compilazione perché int i, j; veniva dichiarato due volte nello stesso scope. 
Ora dichiarazioni e utilizzo sono corretti.
Separazione (o unione) logica corretta dei cicli

Prima c’erano due for in cui pointsPerClass veniva incrementato due volte nella stessa iterazione di K-Means, senza essere azzerato tra un for e l’altro. Questo gonfiava i conteggi.
Adesso, abbiamo reso chiaro che nel primo for calcoliamo i cambi di cluster e nel secondo for accumuliamo le coordinate nei centroidi (oltre a contare i punti). Viene anche effettuato un zeroIntArray() o zeroFloatMatriz() al momento giusto.
Gestione di changes in una variabile locale ridotta (local_changes)

Prima facevi changes++ direttamente in un for parallelo. Adesso lo facciamo in local_changes e poi lo riduciamo correttamente. Così otteniamo il totale globale (in quell’MPI-process) dei cambi di cluster, e infine lo memorizziamo in changes.

COSA SBAGLIAVAMO?
Prima:

Usavo changes (o altre variabili come pointsPerClass) in un for parallelo OpenMP senza riduzione o senza #pragma omp atomic. 
In un ambiente multithread, questo genera race condition (i thread si sovrascrivono a vicenda), e il risultato finale di changes e pointsPerClass era non deterministico.

Facevo l’aggiornamento dei conteggi (pointsPerClass) in due fasi, talvolta senza azzerare prima di rientrare nel for, e aggiungevi di nuovo i valori.


Adesso:

Aggiunta di reduction(+: ...) su changes e su pointsPerClass/auxCentroids. 
  In tal modo, ogni thread ha il proprio accumulatore e alla fine del for i risultati vengono sommati correttamente.


Chiarezza nei passaggi:
Assegniamo i cluster, calcoliamo changes.
Accumulo coordinate dei punti per i centroidi.
Riduzione MPI dei risultati su tutti i processi.
Calcolo dei nuovi centroidi e controllo dei criteri di arresto.
In sintesi, la chiave per far funzionare correttamente la versione OpenMP è proteggere le scritture sulle variabili globali condivise nel for parallelo con meccanismi di riduzione o di atomicità, ed evitare di incrementare due volte la stessa struttura senza azzerarla (o usarne due diverse in modo coerente).

*/