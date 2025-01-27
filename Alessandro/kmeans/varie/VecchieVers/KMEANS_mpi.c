/*
 * k-Means clustering algorithm
 *
 * MPI version (ottimizzata: rimozione sqrt in assegnazione, single sqrt per
 * maxDist)
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.2
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
#include <time.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
Function showFileError: stampa un messaggio di errore in base al valore di
"error"
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
Function readInput: Conta righe (lines) e colonne (samples) del file
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
      // Se la riga eccede MAXLINE, manca '\n' => errore
      if (strchr(line, '\n') == NULL) {
        fclose(fp);
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
Function readInput2: carica effettivamente i dati nel buffer 'data'
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
Function writeResult: scrive su file i cluster dei punti
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
    return -3;
  }
}

/*
Function initCentroids: Copia i dati di alcuni punti casuali come centroidi
iniziali
*/
void initCentroids(const float *data, float *centroids, int *centroidPos,
                   int samples, int K) {
  for (int i = 0; i < K; i++) {
    int idx = centroidPos[i];
    memcpy(&centroids[i * samples], &data[idx * samples],
           (samples * sizeof(float)));
  }
}

/*
Funzione per il calcolo del quadrato della distanza euclidea tra point e center
(Eliminiamo la sqrt, utile durante l'assegnamento)
*/
static inline float distanceSquared(const float *point, const float *center,
                                    int samples) {
  float dist2 = 0.0f;
  for (int i = 0; i < samples; i++) {
    float tmp = point[i] - center[i];
    dist2 += tmp * tmp;
  }
  return dist2;
}

/*
Funzione per calcolare la distanza effettiva (inclusa sqrt),
usata solo quando serve valutare lo spostamento dei centroidi in rank 0
*/
static inline float euclideanDistance(const float *p, const float *c,
                                      int samples) {
  float dist2 = distanceSquared(p, c, samples);
  return sqrtf(dist2);
}

/*
Funzione zeroFloatMatriz: mette a zero una matrice float K x samples
*/
void zeroFloatMatriz(float *matrix, int rows, int columns) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
      matrix[i * columns + j] = 0.0f;
}

/*
Funzione zeroIntArray: mette a zero un array int di dimensione size
*/
void zeroIntArray(int *array, int size) {
  for (int i = 0; i < size; i++)
    array[i] = 0;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  double start, end;
  start = MPI_Wtime();

  // Controllo parametri
  if (argc != 7) {
    if (rank == 0) {
      fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
      fprintf(stderr, "Usage: mpirun -np <procs> ./KMEANS <InputFile> <K> "
                      "<MaxIter> <Min%%Change> <Threshold> <OutputFile>\n");
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Lettura dimensioni (rank 0) e broadcast
  int lines = 0, samples = 0;
  if (rank == 0) {
    int error = readInput(argv[1], &lines, &samples);
    if (error != 0) {
      showFileError(error, argv[1]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Parametri: K, maxIter, minChanges, maxThreshold
  int K, maxIterations, minChanges;
  float maxThreshold;

  K = atoi(argv[2]);
  maxIterations = atoi(argv[3]);
  minChanges = (int)(lines * atof(argv[4]) / 100.0f);
  maxThreshold = atof(argv[5]);

  // Check base
  if (K > lines || K <= 0) {
    fprintf(stderr, "ERROR: Invalid number of clusters (K=%d)!\n", K);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Allocazione array data
  float *data = (float *)malloc(lines * samples * sizeof(float));
  if (!data) {
    fprintf(stderr, "Memory allocation error (data).\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Legge i dati

  int error = readInput2(argv[1], data);
  if (error != 0) {
    showFileError(error, argv[1]);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Allocazione centroidi
  float *centroids = (float *)calloc(K * samples, sizeof(float));
  if (!centroids) {
    fprintf(stderr, "Memory allocation error (centroids).\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Inizializzazione centroidi (solo rank 0)
  int *centroidPos = NULL;
  int *classMapGlobal = NULL;

  centroidPos = (int *)malloc(K * sizeof(int));
  if (!centroidPos) {
    fprintf(stderr, "Memory allocation error (centroidPos).\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  srand(0);
  for (int i = 0; i < K; i++)
    centroidPos[i] = rand() % lines;

  initCentroids(data, centroids, centroidPos, samples, K);

  classMapGlobal = (int *)calloc(lines, sizeof(int));
  if (!classMapGlobal) {
    fprintf(stderr, "Memory allocation error (classMapGlobal).\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Suddivisione del lavoro
  int localSize = lines / numProcs;
  int remainder = lines % numProcs;
  int startIndex = (rank < remainder) ? rank * (localSize + 1)
                                      : remainder * (localSize + 1) +
                                            (rank - remainder) * localSize;
  int endIndex = (rank < remainder) ? startIndex + (localSize + 1)
                                    : startIndex + localSize;
  if (endIndex > lines)
    endIndex = lines;

  int myNumPoints = endIndex - startIndex;

  // Allocazione strutture locali
  int *classMapLocal = (int *)calloc(myNumPoints, sizeof(int));
  int *pointsPerClassLocal = (int *)malloc(K * sizeof(int));
  float *auxCentroidsLocal = (float *)malloc(K * samples * sizeof(float));
  if (!classMapLocal || !pointsPerClassLocal || !auxCentroidsLocal) {
    fprintf(stderr, "Memory allocation error (local structures).\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Per calcolo spostamento centroidi: manteniamo in rank 0 i "oldCentroids"
  float *oldCentroids = NULL;
  if (rank == 0) {
    oldCentroids = (float *)malloc(K * samples * sizeof(float));
    if (!oldCentroids) {
      fprintf(stderr, "Memory allocation error (oldCentroids).\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Per output in rank 0
  char *outputMsg = NULL;
  if (rank == 0) {
    outputMsg = (char *)calloc(10000, sizeof(char));
    if (!outputMsg) {
      fprintf(stderr, "Memory allocation error (outputMsg).\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1],
           lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum iterations: %d\n", maxIterations);
    printf("\tMinimum changes: %d [%.2f%% of %d points]\n", minChanges,
           atof(argv[4]), lines);
    printf("\tMax centroid precision: %f\n", maxThreshold);
  }

  end = MPI_Wtime();
  if (rank == 0) {
    printf("\nInitialization: %f seconds\n", end - start);
    fflush(stdout);
  }

  // MAIN LOOP k-means
  start = MPI_Wtime();
  int it = 0;
  int globalChanges = 0;
  float maxDist = 0.0f;
  int continueFlag = 1;

  do {
    it++;

    // Azzeriamo accumuli locali
    zeroIntArray(pointsPerClassLocal, K);
    zeroFloatMatriz(auxCentroidsLocal, K, samples);

    // 1. Assegnamento cluster (usiamo distanceSquared)
    int localChanges = 0;
    for (int i = 0; i < myNumPoints; i++) {
      int realIndex = startIndex + i;
      float minDist2 = FLT_MAX;
      int bestClass = 1;

      // Trova cluster con distanza quadratica minima
      for (int j = 0; j < K; j++) {
        float dist2 = distanceSquared(&data[realIndex * samples],
                                      &centroids[j * samples], samples);
        if (dist2 < minDist2) {
          minDist2 = dist2;
          bestClass = j + 1;
        }
      }
      if (classMapLocal[i] != bestClass) {
        localChanges++;
      }
      classMapLocal[i] = bestClass;

      // Accumulo per calcolo medie
      int cidx = bestClass - 1;
      pointsPerClassLocal[cidx]++;
      for (int d = 0; d < samples; d++) {
        auxCentroidsLocal[cidx * samples + d] += data[realIndex * samples + d];
      }
    }

    // Riduzione del numero totale di cambi cluster
    MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    // Riduzione per pointsPerClass e auxCentroids
    // useremo buffer temporanei in rank 0
    int *globalPointsPerClass = NULL;
    float *globalAuxCentroids = NULL;
    if (rank == 0) {
      globalPointsPerClass = (int *)calloc(K, sizeof(int));
      globalAuxCentroids = (float *)calloc(K * samples, sizeof(float));
      if (!globalPointsPerClass || !globalAuxCentroids) {
        fprintf(stderr, "Memory allocation error (globalPointsPerClass / "
                        "globalAuxCentroids).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      }
    }

    MPI_Reduce(pointsPerClassLocal, (rank == 0 ? globalPointsPerClass : NULL),
               K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(auxCentroidsLocal, (rank == 0 ? globalAuxCentroids : NULL),
               K * samples, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // 2. (solo rank 0) Aggiorniamo i centroidi e calcoliamo maxDist
    if (rank == 0) {
      // Copiamo centroidi vecchi
      memcpy(oldCentroids, centroids, K * samples * sizeof(float));

      // Nuovi centroidi = media
      for (int c = 0; c < K; c++) {
        int countC = globalPointsPerClass[c];
        if (countC > 0) {
          for (int d = 0; d < samples; d++) {
            globalAuxCentroids[c * samples + d] /= (float)countC;
          }
        }
      }

      // Calcoliamo spostamento max (usiamo euclideanDistance con sqrt, ma solo
      // su centroidi)
      maxDist = 0.0f;
      for (int c = 0; c < K; c++) {
        float distC =
            euclideanDistance(&oldCentroids[c * samples],
                              &globalAuxCentroids[c * samples], samples);
        if (distC > maxDist)
          maxDist = distC;
      }

      // Aggiorniamo i centroidi
      memcpy(centroids, globalAuxCentroids, K * samples * sizeof(float));

      // Messaggi di log
      char buffer[128];
      sprintf(buffer, "\n[%d] Changes: %d\tMaxDist: %f", it, globalChanges,
              maxDist);
      strcat(outputMsg, buffer);

      // Verifica condizione di terminazione
      if ((globalChanges <= minChanges) || (it >= maxIterations) ||
          (maxDist <= maxThreshold))
        continueFlag = 0;

      free(globalPointsPerClass);
      free(globalAuxCentroids);
    }

    // 3. Broadcast continueFlag e, se si continua, i centroidi aggiornati
    MPI_Bcast(&continueFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (continueFlag) {
      MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

  } while (continueFlag);

  // 4. Raccolta classMapLocal in rank 0
  int *recvCounts = NULL;
  int *displs = NULL;
  if (rank == 0) {
    recvCounts = (int *)malloc(numProcs * sizeof(int));
    displs = (int *)malloc(numProcs * sizeof(int));
    if (!recvCounts || !displs) {
      fprintf(stderr, "Memory allocation error (recvCounts/displs).\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  MPI_Gather(&myNumPoints, 1, MPI_INT, recvCounts, 1, MPI_INT, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    displs[0] = 0;
    for (int p = 1; p < numProcs; p++) {
      displs[p] = displs[p - 1] + recvCounts[p - 1];
    }
  }

  MPI_Gatherv(classMapLocal, myNumPoints, MPI_INT, classMapGlobal, recvCounts,
              displs, MPI_INT, 0, MPI_COMM_WORLD);

  // Fine calcolo
  end = MPI_Wtime();
  float comp_time = end - start;
  MPI_Reduce(&comp_time, MPI_IN_PLACE, 1, MPI_FLOAT, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nComputation: %f seconds", comp_time);
    fflush(stdout);
  }
  if (rank == 0) {
    // Stampa log
    printf("%s", outputMsg);
    // Cause di terminazione
    if (globalChanges <= minChanges)
      printf("\n\nTermination condition:\nMinimum number of changes reached: "
             "%d [%d]",
             globalChanges, minChanges);
    else if (it >= maxIterations)
      printf("\n\nTermination condition:\nMaximum number of iterations "
             "reached: %d [%d]",
             it, maxIterations);
    else
      printf("\n\nTermination condition:\nCentroid update precision reached: "
             "%g [%g]",
             maxDist, maxThreshold);

    // Scrittura su file
    int error = writeResult(classMapGlobal, lines, argv[6]);
    if (error != 0) {
      showFileError(error, argv[6]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Cleanup
  free(data);
  free(centroids);
  free(classMapLocal);
  free(pointsPerClassLocal);
  free(auxCentroidsLocal);
  if (rank == 0) {
    free(centroidPos);
    free(classMapGlobal);
    free(outputMsg);
    free(recvCounts);
    free(displs);
    free(oldCentroids);
  }

  MPI_Finalize();
  return 0;
}
