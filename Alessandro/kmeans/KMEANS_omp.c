/*
 * k-Means clustering algorithm
 *
 * OpenMP version
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
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXLINE 2000
#define MAXCAD 200
#define UNROLL 2

#define CALLTIME(start, call)                                                  \
  call;                                                                        \
  double end_tmp = omp_get_wtime();                                            \
  printf("\n%s time : %lf\n ", #call, (end_tmp - start));
// Misura il tempo di esecuzione di una singola chiamata o funzione

#define BLOCKTIME(start, block)                                                \
  block double end_tmp = omp_get_wtime();                                      \
  printf("\n%s time : %lf\n ", #block, (end_tmp - start));

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
Function showFileError: It displays the corresponding error during file reading.
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
  // Cliclo esterno per avanzare nei blocchi

  for (i = 0; i < samples; i += blockSize) {
    float blockDist = 0.0f; // Accumulatore temporaneo per il blocco
    // Ciclo interno per calcolare la distanza tra i punti
    for (j = i; j < i + blockSize && j < samples; j++) {
      diff = (point[j] - center[j]) * (point[j] - center[j]);
      blockDist += diff;
    }
    dist += blockDist; // Somma il risultato del blocco
  }

  return dist; // Restituisce la distanza al quadrato
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < columns; j++)
      matrix[i * columns + j] = 0.0;
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size) {
  int i;
  for (i = 0; i < size; i++)
    array[i] = 0;
}

int main(int argc, char *argv[]) {

  // START CLOCK***************************************
  double start, end;
  start = omp_get_wtime();
  //**************************************************
  /*
   * PARAMETERS
   *
   * argv[1]: Input data file
   * argv[2]: Number of clusters
   * argv[3]: Maximum number of iterations of the method. Algorithm termination
   * condition. argv[4]: Minimum percentage of class changes. Algorithm
   * termination condition. If between one iteration and the next, the
   * percentage of class changes is less than this percentage, the algorithm
   * stops. argv[5]: Precision in the centroid distance after the update. It is
   * an algorithm termination condition. If between one iteration of the
   * algorithm and the next, the maximum distance between centroids is less than
   * this precision, the algorithm stops. argv[6]: Output file. Class assigned
   * to each point of the input file.
   * */
  if (argc != 7) {
    fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
    fprintf(stderr,
            "./KMEANS [Input Filename] [Number of clusters] [Number of "
            "iterations] [Number of changes] [Threshold] [Output data file]\n");
    fflush(stderr);
    exit(-1);
  }

  // Reading the input data
  // lines = number of points; samples = number of dimensions per point
  int lines = 0, samples = 0;

  int error = readInput(argv[1], &lines, &samples);
  if (error != 0) {
    showFileError(error, argv[1]);
    exit(error);
  }

  float *data = (float *)calloc(lines * samples, sizeof(float));
  if (data == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    exit(-4);
  }
  error = readInput2(argv[1], data);
  if (error != 0) {
    showFileError(error, argv[1]);
    exit(error);
  }

  // Parameters
  int K = atoi(argv[2]);
  int maxIterations = atoi(argv[3]);
  int minChanges = (int)(lines * atof(argv[4]) / 100.0);
  float maxThreshold = atof(argv[5]);
/*
centroidPos: Indici dei punti del dataset scelti come centroidi iniziali.
centroids: Coordinate dei centroidi correnti (inizializzate a 0 ma poi popolate con initCentroids).
classMap: Mappa che associa ogni punto del dataset al suo cluster corrente.
*/
  int *centroidPos = (int *)calloc(K, sizeof(int));
  float *centroids = (float *)calloc(K * samples, sizeof(float));
  int *classMap = (int *)calloc(lines, sizeof(int));

  if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    exit(-4);
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
  end = omp_get_wtime();
  printf("\nMemory allocation: %f seconds\n", end - start);
  fflush(stdout);
  //**************************************************
  // START CLOCK***************************************
  start = omp_get_wtime();
  //**************************************************
  char *outputMsg = (char *)calloc(10000, sizeof(char));

  int class;
  int it = 0;
  int changes = 0;
  float maxDist;

  // pointPerClass: Conta il numero di punti assegnati a ciascun cluster.
  // auxCentroids: Contiene le somme delle coordinate dei punti assegnati a ciascun cluster, utilizzato per calcolare la media (centroide) di ciascun cluster aggiornandolo.
  // distCentroids: Memorizza le distanze tra i centroidi precedenti e quelli aggiornati nell'iterazione corrente, calcolare il criterio di precisione dell'algoritmo, ovvero se i centroidi si sono spostati sotto una certa soglia (maxThreshold), il K-means può terminare.
  int *pointsPerClass = (int *)malloc(K * sizeof(int));
  float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
  float *distCentroids = (float *)malloc(K * sizeof(float));
  if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    exit(-4);
  }

  /*
   *
   * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
   *
   */
  // Decido quale funzione usare in base all input, 2 dimensioni -> UNROLLEDeuclideanDistance, altrimenti euclideanDistance
  float (*distanceFun)(float *, float *, int) =
      (samples % 2 == 0) ? UNROLLEDeuclideanDistance : euclideanDistance;
  do {
    it++;
    changes = 0;
//Questo blocco di codice implementa la fase di assegnazione dei punti ai centroidi più vicini nel k-means
/*
La direttiva #pragma omp parallel for viene usata per parallelizzare il ciclo esterno (for i = 0), che iterava sequenzialmente nella versione seriale. 
In particolare:
- La clausola reduction(+ : changes) viene utilizzata per garantire che la variabile changes sia aggiornata correttamente in parallelo.
- La clausola schedule(static) viene utilizzata per distribuire equamente il lavoro tra i thread, utilizziamo static invece che dynamic per evitare overhead di sincronizzazione e perche i dati sono regolari.
- La variabile class è dichiarata privata per garantire che ogni thread abbia la propria copia locale.
*/
#pragma omp parallel for reduction(+ : changes) schedule(guided) private(class)
    for (int i = 0; i < lines; i++) {
      class = 1;
      float minDist = FLT_MAX;
      for (int j = 0; j < K; j++) {
        float dist =
            distanceFun(&data[i * samples], &centroids[j * samples], samples);
        if (dist < minDist) {
          minDist = dist;
          class = j + 1;
        }
      }
      if (classMap[i] != class) {
        changes++;
      }
      classMap[i] = class;
    }
    zeroFloatMatriz(auxCentroids, K, samples);
    for (int i = 0; i < lines; i++) {
      class = classMap[i] - 1;
      pointsPerClass[class]++;
      for (int j = 0; j < samples; j++) {
        auxCentroids[class * samples + j] += data[i * samples + j];
      }
    }


// Questo blocco di codice aggiorna i centroidi calcolando la media dei punti appartenenti a ciascun cluster.
/*
Con OpenMP, più thread aggiornano simultaneamente i centroidi di diversi cluster.
La direttiva #pragma omp parallel for viene utilizzata per parallelizzare il ciclo esterno (for i = 0), che iterava sequenzialmente nella versione seriale.
In particolare:
- La clausola schedule(static) viene utilizzata per distribuire equamente il lavoro tra i thread, utilizziamo static invece che dynamic per evitare overhead di sincronizzazione e perche i dati sono regolari.
- La variabile i è dichiarata privata per garantire che ogni thread abbia la propria copia locale.
*/  
#pragma omp parallel for schedule(static)
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < samples; j++) {
        if (pointsPerClass[i] > 0) {
          auxCentroids[i * samples + j] /= pointsPerClass[i];
        }
      }
    }
    zeroIntArray(pointsPerClass, K);

    maxDist = FLT_MIN;

// Questo blocco di codice calcola la distanza tra i centroidi precedenti e quelli aggiornati nell'iterazione corrente.
/*
Con OpenMP, più thread calcolano simultaneamente le distanze tra i centroidi di diversi cluster.
La direttiva #pragma omp parallel for viene utilizzata per parallelizzare il ciclo esterno (for i = 0), che iterava sequenzialmente nella versione seriale.
In particolare:
- La clausola reduction(max : maxDist) viene utilizzata per garantire che la variabile maxDist sia aggiornata correttamente in parallelo.
- La clausola schedule(static) viene utilizzata per distribuire equamente il lavoro tra i thread, utilizziamo static invece che dynamic per evitare overhead di sincronizzazione e perche i dati sono regolari.
- La variabile i è dichiarata privata per garantire che ogni thread abbia la propria copia locale.
*/ 
#pragma omp parallel for reduction(max : maxDist) schedule(guided)
    for (int i = 0; i < K; i++) {
      distCentroids[i] = distanceFun(&centroids[i * samples],
                                     &auxCentroids[i * samples], samples);
      maxDist = MAX(maxDist, distCentroids[i]);
    }

    memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

    printf("[%d] Cluster changes: %d\tMax. centroid distance: %f\n", it,
           changes, sqrt(maxDist));

  } while ((changes > minChanges) && (it < maxIterations) &&
           (sqrt(maxDist) > maxThreshold));

  /*
   *
   * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
   *
   */
  // Output and termination conditions
  printf("%s", outputMsg);

  // END CLOCK*****************************************
  end = omp_get_wtime();
  printf("\nComputation: %f seconds", end - start);
  fflush(stdout);
  //**************************************************
  // START CLOCK***************************************
  start = omp_get_wtime();
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

  // Writing the classification of each point to the output file.
  error = writeResult(classMap, lines, argv[6]);
  if (error != 0) {
    showFileError(error, argv[6]);
    exit(error);
  }

  // Free memory
  free(data);
  free(classMap);
  free(centroidPos);
  free(centroids);
  free(distCentroids);
  free(pointsPerClass);
  free(auxCentroids);

  // END CLOCK*****************************************
  end = omp_get_wtime();
  printf("\n\nMemory deallocation: %f seconds\n", end - start);
  fflush(stdout);
  //***************************************************/
  return 0;
}
