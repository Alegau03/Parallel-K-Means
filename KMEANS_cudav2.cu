/*
 * k-Means clustering algorithm
 *
 * CUDA version
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
#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// NOTE : compile with flag -Xptxas -dlcm=cg to disable L1 cache
/*profile tools : -nsys profile ./a.out args...
                  -nvprof ./app args ...
                  -nsys-ui
                  -ncu */
#define UNROLL 2
#define NUM_WARP_SCHEDULERS 4
#define POINTS_PER_THREAD 4
#define REG_PER_THREAD 32 // nvcc -Xcompiler -fopenmp -Xptxas -v  KMEANS_cuda.cu
#define MAX_BLOCK_DIM 1024
#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
 */
#define CHECK_CUDA_CALL(a)                                                     \
  {                                                                            \
    cudaError_t ok = a;                                                        \
    if (ok != cudaSuccess)                                                     \
      fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__,         \
              cudaGetErrorString(ok));                                         \
  }
#define CHECK_CUDA_LAST()                                                      \
  {                                                                            \
    cudaError_t ok = cudaGetLastError();                                       \
    if (ok != cudaSuccess)                                                     \
      fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__,         \
              cudaGetErrorString(ok));                                         \
  }

__constant__ int d_K, d_samples,
    d_lines; // numero di cluster, dimensioni, numero di punti memorizzati tutti
             // nella costant memory e utilizzati da tutti i thread
/*Segnatura di GPU_ClassAssignment e GPU_CentroidsUpdate */
__global__ void GPU_ClassAssignment(int *d_pointsPerClass,
                                    float *d_auxCentroids, int *d_changes,
                                    float *d_centroids, float *d_data,
                                    int *d_classMap);
__global__ void GPU_CentroidsUpdate(int *d_pointsPerClass,
                                    float *d_auxCentroids, float *d_centroids);
void OptimalBlockGridDims(int numberOfThreads, int *optBlockDim,
                          int *optGridDim);
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
int writeResult(int *ClassMap, int lines, const char *filename) {
  FILE *fp;

  if ((fp = fopen(filename, "wt")) != NULL) {
    for (int i = 0; i < lines; i++) {
      fprintf(fp, "%d\n", ClassMap[i]);
    }
    fclose(fp);

    return 0;
  } else {
    return -3; // No file found
  }
}

/*

Function initCentroids: This function copies the values of the initial
centroids, using their global_id in the input data structure as a reference map.
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
Function euclideanDistance: Euclidean distance
This function could be modified
*/
/* Modifica della distanza euclidea con l'unroll come nelle altre
 * implementazioni, inoltre la funzione può essere utilizzata sia da device che
 * da host */
__device__ __host__ float euclideanDistance(float *point, float *center,
                                            int samples) {
  float dist1 = 0.0;
  float dist2 = 0.0;
  for (int i = 0; i * UNROLL < samples;
       i++) { // Unroll di due iterazioni del ciclo for
    dist1 += (point[i] - center[i]) * (point[i] - center[i]);
    dist2 += (point[i + 1] - center[i + 1]) * (point[i + 1] - center[i + 1]);
  }
  return (dist1 + dist2);
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
   * condition. argv[4]: Minimum percentage of Class changes. Algorithm
   * termination condition. If between one iteration and the next, the
   * percentage of Class changes is less than this percentage, the algorithm
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

  int *centroidPos = (int *)calloc(K, sizeof(int));
  float *centroids = (float *)calloc(K * samples, sizeof(float));
  int *ClassMap = (int *)calloc(lines, sizeof(int));

  if (centroidPos == NULL || centroids == NULL || ClassMap == NULL) {
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

  CHECK_CUDA_CALL(cudaSetDevice(0));
  CHECK_CUDA_CALL(cudaDeviceSynchronize());
  //**************************************************
  // START CLOCK***************************************
  start = omp_get_wtime();
  //**************************************************
  char *outputMsg = (char *)calloc(10000, sizeof(char));
  char line[100];

  int it = 0;
  int changes = 0;
  float maxDist;

  // pointPerClass: number of points Classified in each Class
  // auxCentroids: mean of the points in each Class
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
  int gridSize = 0, blockSize = 0;
  int *d_pointsPerClass, *d_changes, *d_classMap;
  float *d_centroids, *d_auxCentroids, *d_data;

  OptimalBlockGridDims(
      ceil(lines / POINTS_PER_THREAD), &blockSize,
      &gridSize); // calcolo della dimensione ottimale di griglia e blocco in
                  // base al numero di punti da classficare

  /*Allocazione di centroid, data, auxCentroids, pointsPerClass, changes e
   * classMap in memoria globale del dispositivo*/
  CHECK_CUDA_CALL(
      cudaMalloc((void **)&d_centroids, K * samples * sizeof(float)));
  CHECK_CUDA_CALL(
      cudaMalloc((void **)&d_data, lines * samples * sizeof(float)));
  CHECK_CUDA_CALL(cudaMalloc((void **)&d_classMap, lines * sizeof(int)));
  CHECK_CUDA_CALL(
      cudaMalloc((void **)&d_auxCentroids, K * samples * sizeof(float)));
  CHECK_CUDA_CALL(cudaMalloc((void **)&d_pointsPerClass, K * sizeof(int)));
  CHECK_CUDA_CALL(cudaMalloc((void **)&d_changes, sizeof(int)));

  /*Copia dei centroidi, dati e mappa delle classi da host a device*/
  CHECK_CUDA_CALL(cudaMemcpy(d_centroids, centroids,
                             K * samples * sizeof(float),
                             cudaMemcpyHostToDevice));
  CHECK_CUDA_CALL(cudaMemcpy(d_data, data, lines * samples * sizeof(float),
                             cudaMemcpyHostToDevice));
  CHECK_CUDA_CALL(cudaMemcpy(d_classMap, ClassMap, lines * sizeof(int),
                             cudaMemcpyHostToDevice));

  /*Azzeramento degli array di dati allocati sul device*/
  CHECK_CUDA_CALL(cudaMemset(d_auxCentroids, .0, K * samples * sizeof(float)));
  CHECK_CUDA_CALL(cudaMemset(d_pointsPerClass, 0, K * sizeof(int)));
  CHECK_CUDA_CALL(cudaMemset(d_changes, 0, sizeof(int)));

  /* Copia dei dati che risiedono in costant memory  da host a device */
  CHECK_CUDA_CALL(cudaMemcpyToSymbol(d_lines, &lines, sizeof(int), 0,
                                     cudaMemcpyHostToDevice));
  CHECK_CUDA_CALL(
      cudaMemcpyToSymbol(d_K, &K, sizeof(int), 0, cudaMemcpyHostToDevice));
  CHECK_CUDA_CALL(cudaMemcpyToSymbol(d_samples, &samples, sizeof(int), 0,
                                     cudaMemcpyHostToDevice));
  do {
    // azzeramento dei cambiamenti nel cluster ad ogni iterazione
    changes = 0;
    /* Lancio del kernel  in cui si classificano i punti in d_data e si
     * accumulano coordinate in auxCentroids, e numero di punti per classe in
     * pointsPerClass*/
    GPU_ClassAssignment<<<gridSize, blockSize>>>(
        d_pointsPerClass, d_auxCentroids, d_changes, d_centroids, d_data,
        d_classMap);
    cudaDeviceSynchronize(); // attende la fine del kernel da parte di ogni
                             // thread
    CHECK_CUDA_LAST();

    /*Copia dei cambiament i da device ad host dopo la sincronizzazione*/
    CHECK_CUDA_CALL(
        cudaMemcpy(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost));

    /*Aggiornamento delle coordinate in d_centroids*/
    GPU_CentroidsUpdate<<<K, samples>>>(d_pointsPerClass, d_auxCentroids,
                                        d_centroids);
    cudaDeviceSynchronize(); // attende la fine del kernel da parte di ogni
                             // thread
    CHECK_CUDA_LAST();
    /* Copia dei centroidi d_centroids in auxCentroids*/
    CHECK_CUDA_CALL(cudaMemcpy(auxCentroids, d_centroids,
                               K * samples * sizeof(float),
                               cudaMemcpyDeviceToHost));

    /* calcolo della distanza di aggiornamento maggiore*/
    maxDist = FLT_MIN;
    for (i = 0; i < K; i++) {
      distCentroids[i] = euclideanDistance(&centroids[i * samples],
                                           &auxCentroids[i * samples], samples);
      if (distCentroids[i] > maxDist) {
        maxDist = distCentroids[i];
      }
    }

    memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

    sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it,
            changes, sqrt(maxDist));
    outputMsg = strcat(outputMsg, line);

    // reset degli array ausiliari sul device e di d_changes
    CHECK_CUDA_CALL(
        cudaMemset(d_auxCentroids, .0, K * samples * sizeof(float)));
    CHECK_CUDA_CALL(cudaMemset(d_pointsPerClass, 0, K * sizeof(float)));
    CHECK_CUDA_CALL(cudaMemset(d_changes, 0, sizeof(int)));

  } while ((changes > minChanges) && (it < maxIterations) &&
           (maxDist > maxThreshold * maxThreshold));

  // copia finale della mappa delle classi sul device all'host
  CHECK_CUDA_CALL(cudaMemcpy(ClassMap, d_classMap, lines * sizeof(int),
                             cudaMemcpyDeviceToHost));
  /*
   *
   * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
   *
   */
  // Output and termination conditions
  printf("%s", outputMsg);

  CHECK_CUDA_CALL(cudaDeviceSynchronize());

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

  // Writing the Classification of each point to the output file.
  error = writeResult(ClassMap, lines, argv[6]);
  if (error != 0) {
    showFileError(error, argv[6]);
    exit(error);
  }

  // Free memory
  free(data);
  free(ClassMap);
  free(centroidPos);
  free(centroids);
  free(distCentroids);
  free(pointsPerClass);
  free(auxCentroids);
  CHECK_CUDA_CALL(cudaFree(d_auxCentroids));
  CHECK_CUDA_CALL(cudaFree(d_data));
  CHECK_CUDA_CALL(cudaFree(d_centroids));
  CHECK_CUDA_CALL(cudaFree(d_changes));
  CHECK_CUDA_CALL(cudaFree(d_pointsPerClass));
  CHECK_CUDA_CALL(cudaFree(d_classMap));

  // END CLOCK*****************************************
  end = omp_get_wtime();
  printf("\n\nMemory deallocation: %f seconds\n", end - start);
  fflush(stdout);
  //***************************************************/
  return 0;
}
/*Funzione per il calcolo delle dimensioni ideali di griglia e blocco:
in:
        - numero di thread da creare -> int numberOfThreads
out:
        - indirizzo in cui salvare la dimensione ottimale del blocco -> int
*optBlockDim
        - indirizzo in cui salvare la dimensione ottimale della griglia -> int
*optGridDim */
void OptimalBlockGridDims(int numberOfThreads, int *optBlockDim,
                          int *optGridDim) {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);

  int maxRegs = p.regsPerBlock;
  int maxThreadsPerSM = p.maxThreadsPerMultiProcessor;
  int GridDim = p.multiProcessorCount;
  int BlockDim = p.warpSize;
  while (GridDim * BlockDim < numberOfThreads) {
    if ((BlockDim + 2 * p.warpSize) <
        MIN(maxThreadsPerSM, maxRegs / REG_PER_THREAD)) {
      BlockDim += p.warpSize;
    }

    GridDim += (p.multiProcessorCount / 2);
  }
  *optGridDim = GridDim;
  *optBlockDim = BlockDim;

  return;
}

/* Funzione che tiene conto dell'imbalance
 void OptimalBlockGridDims(int numberOfThreads, int *optBlockDim,
                           int *optGridDim) {
   cudaDeviceProp p;
   cudaGetDeviceProperties(&p, 0);

   int maxRegs = p.regsPerBlock;
   int maxThreadsPerSM = p.maxThreadsPerMultiProcessor;
   int SM = p.multiProcessorCount;
   int warp = p.warpSize;
   int bestImbalance = SM, imbalance;

   int threadsPerBlock = MIN(maxThreadsPerSM, maxRegs / REG_PER_THREAD);

   int tmp = threadsPerBlock / warp;
   int totalBlocks;
   threadsPerBlock = (tmp + 1) * warp;

   for (; threadsPerBlock >= NUM_WARP_SCHEDULERS * warp && bestImbalance != 0;
        threadsPerBlock -= warp) {
     totalBlocks = (int)ceil(1.0 * numberOfThreads / threadsPerBlock);
     if (totalBlocks % SM == 0) {
       imbalance = 0;
     } else {
       int blocksPerSM = totalBlocks / SM;
       imbalance = (SM - (totalBlocks % SM)) / (blocksPerSM + 1.0);
     }
     if (bestImbalance >= imbalance) {
       bestImbalance = imbalance;
       *optGridDim = totalBlocks;
       *optBlockDim = threadsPerBlock;
     }
   }
 }*/

/*Funzione che classifica i punti in d_data e accumula le loro coordinate nelle
 * coordinate dei centroidi a cui appartengono in d_auxCentroids.
 in: -array di punti per class -> int *d_pointsPerClass
     -array che accumula le coordinate dei centroidi -> float* d_auxCentroids
     -contatore dei cambiamenti nei cluster -> int *d_changes
     -array di coordinate dei centroidi -> float *d_centroids
     -array di tutti i punti del dataset -> float *d_data
     -array contente la mappa delle class -> int *d_classMap*/
__global__ void GPU_ClassAssignment(int *d_pointsPerClass,
                                    float *d_auxCentroids, int *d_changes,
                                    float *d_centroids, float *d_data,
                                    int *d_classMap) {
  int global_id =
      threadIdx.x + blockIdx.x * blockDim.x; // calcolo della posizione del
                                             // thread all'interno della griglia
  int local_changes = 0; // contatore dei cambiamenti locali : può andare da 0 a
                         // POINTS_PER_THREAD

  // per ogni punto classifico POINTS_PER_THREAD punti
  for (int i = 0; i < POINTS_PER_THREAD; ++i) {
    int pointIndex = (global_id * POINTS_PER_THREAD) +
                     i; // calcolo della posizione del punto da classificare
                        // all'interno del dataset
    if (pointIndex >= d_lines) // se sforo l'array non faccio nulla
      return;
    /* questa parte è identica alla versione seriale tranne per l'unroll nella
     * distanza euclidea */
    int Class = 1;
    float minDist = FLT_MAX;
    float dist;
    for (int j = 0; j < d_K; j++) {
      dist = euclideanDistance(&d_data[pointIndex * d_samples],
                               &d_centroids[j * d_samples], d_samples);
      if (dist < minDist) {
        minDist = dist;
        Class = j + 1;
      }
    }
    if (d_classMap[pointIndex] !=
        Class) { // aggiorno d_classMap solo quando è necessario
      d_classMap[pointIndex] = Class;
      local_changes++; // aggiorno i cambiamenti locali
    }

    Class -= 1;
    atomicAdd(&d_pointsPerClass[Class],
              1); // aggiorno in modo sicuro pointsPerClass[Class] di 1
    // aggiorno ogni coordinata del centroide con quelle del punto in modo
    // consistente a tutti gli altri thread
    for (int j = 0; j < d_samples; j++) {
      atomicAdd(&d_auxCentroids[Class * d_samples + j],
                d_data[pointIndex * d_samples + j]);
    }
  }
  if (local_changes > 0) // aggiorno i cambiamenti globali in modo consistente
                         // solo se effettivamente ho verificato cambiamenti
    atomicAdd(d_changes, local_changes);
}

__global__ void GPU_PointClassification(float *d_centroids, float *d_data,
                                        int *d_classMap) {
  int global_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadIdx.x >= d_samples)
    return;
  extern __shared__ float distance_vector[];
  float partial_distance, centr_coord, data_coord;
  for (int i = 0; i < d_K; i++) {
    data_coord = d_data[global_id];
    centr_coord = d_centroids[i];
    partial_distance = (data_coord - centr_coord) * (data_coord - centr_coord);
    atomicAdd(&distance_vector[i], partial_distance);
  }
  __syncthreads();
  int Class;
  if (threadIdx.x > floor(d_K / 2))
    return;
  for (int i = 0; i <= (d_K - threadIdx.x) / 2; i++) {
    //....
  }
}

/*Funzione che aggiorna le coordinate dei centroidi in d_centroids
in:  -array di punti per class -> int *d_pointsPerClass
     -array che accumula le coordinate dei centroidi -> float* d_auxCentroids
     -array dei centroidi (da aggiornare) -> float* auxCentroids
Ogni blocco si occupa delle coordinate di un centroide, quindi ogni thread
all'interno di uno stesso blocco, calcola una coordinata del centroide di cui
il blocco si occupa*/
__global__ void GPU_CentroidsUpdate(int *d_pointsPerClass,
                                    float *d_auxCentroids, float *d_centroids) {
  int global_id =
      threadIdx.x +
      blockIdx.x *
          blockDim
              .x; // calcolo della posizione del thread nelle griglia e blocco
  if (global_id > d_samples * d_K)
    return;
  int numPoints =
      d_pointsPerClass[blockIdx.x]; // numero di punti per la classe
                                    // in cui si trova il thread (blockIdx.x)
  if (numPoints == 0)
    return;

  d_centroids[global_id] =
      d_auxCentroids[global_id] /
      numPoints; // calcolo delle coordinate aggiornate dei centroidi
}