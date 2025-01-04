/*
 * k-Means clustering algorithm
 * OpenMP optimized version
 */
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXLINE 2000

#define CALLTIME(start, call)                                                  \
  call;                                                                        \
  double end_tmp = omp_get_wtime();                                            \
  printf("\n%s time : %lf\n ", #call, (end_tmp - start));

#define BLOCKTIME(start, block)                                                \
  block double end_tmp = omp_get_wtime();                                      \
  printf("\n%s time : %lf\n ", #block, (end_tmp - start));
// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Function prototypes */
void showFileError(int error, char *filename);
int readInput(char *filename, int *lines, int *samples);
int readInput2(char *filename, float *data);
int writeResult(int *classMap, int lines, const char *filename);
void initCentroids(const float *data, float *centroids, int *centroidPos,
                   int samples, int K);
float euclideanDistance(float *point, float *center, int samples);
void zeroFloatMatrix(float *matrix, int rows, int columns);
void zeroIntArray(int *array, int size);

int main(int argc, char *argv[]) {
  // Start clock
  double start, end;
  start = omp_get_wtime();

  // Input validation
  if (argc != 7) {
    fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
    fprintf(stderr,
            "./KMEANS [Input Filename] [Number of clusters] [Number of "
            "iterations] [Number of changes] [Threshold] [Output data file]\n");
    exit(-1);
  }

  // Read input data
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
  int *classMap = (int *)calloc(lines, sizeof(int));

  if (centroidPos == NULL || centroids == NULL || classMap == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    exit(-4);
  }

  // Initialize centroids randomly
  srand(0);
  for (int i = 0; i < K; i++) {
    centroidPos[i] = rand() % lines;
  }
  initCentroids(data, centroids, centroidPos, samples, K);

  printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines,
         samples);
  printf("\tNumber of clusters: %d\n", K);
  printf("\tMaximum number of iterations: %d\n", maxIterations);
  printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges,
         atof(argv[4]), lines);
  printf("\tMaximum centroid precision: %f\n", maxThreshold);

  // Allocate auxiliary structures
  int *pointsPerClass = (int *)malloc(K * sizeof(int));
  float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
  float *distCentroids = (float *)malloc(K * sizeof(float));
  if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL) {
    fprintf(stderr, "Memory allocation error.\n");
    exit(-4);
  }

  // Start computation
  int it = 0, changes = 0;
  float maxDist;

  end = omp_get_wtime();
  printf("\nMemory allocation: %f seconds\n", end - start);

  start = omp_get_wtime();

  do {
    it++;
    changes = 0;

// Step 1: Assign points to the nearest centroid
#pragma omp parallel for reduction(+ : changes) schedule(static)
    for (int i = 0; i < lines; i++) {
      int class = -1;
      float minDist = FLT_MAX;
      for (int j = 0; j < K; j++) {
        float dist = euclideanDistance(&data[i * samples],
                                       &centroids[j * samples], samples);
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

    // Step 2: Recalculate centroids
    zeroIntArray(pointsPerClass, K);
    zeroFloatMatrix(auxCentroids, K, samples);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < lines; i++) {
      int class = classMap[i] - 1;
#pragma omp atomic
      pointsPerClass[class]++;

      for (int j = 0; j < samples; j++) {
#pragma omp atomic
        auxCentroids[class * samples + j] += data[i * samples + j];
      }
    }

#pragma omp parallel for schedule(static)
    for (int i = 0; i < K; i++) {
      for (int j = 0; j < samples; j++) {
        if (pointsPerClass[i] > 0) {
          auxCentroids[i * samples + j] /= pointsPerClass[i];
        }
      }
    }

    // Step 3: Calculate maximum distance between old and new centroids
    maxDist = FLT_MIN;
#pragma omp parallel for reduction(max : maxDist) schedule(static)
    for (int i = 0; i < K; i++) {
      distCentroids[i] = euclideanDistance(&centroids[i * samples],
                                           &auxCentroids[i * samples], samples);
      maxDist = MAX(maxDist, distCentroids[i]);
    }

// Update centroids
#pragma omp parallel for schedule(static)
    for (int i = 0; i < K * samples; i++) {
      centroids[i] = auxCentroids[i];
    }

    printf("[%d] Cluster changes: %d\tMax. centroid distance: %f\n", it,
           changes, maxDist);

  } while ((changes > minChanges) && (it < maxIterations) &&
           (maxDist > maxThreshold));

  // Output termination condition
  if (changes <= minChanges) {
    printf("\n\nTermination condition:\nMinimum number of changes reached: %d "
           "[%d]\n",
           changes, minChanges);
  } else if (it >= maxIterations) {
    printf("\n\nTermination condition:\nMaximum number of iterations reached: "
           "%d [%d]\n",
           it, maxIterations);
  } else {
    printf("\n\nTermination condition:\nCentroid update precision reached: %g "
           "[%g]\n",
           maxDist, maxThreshold);
  }

  // Write results
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
  free(pointsPerClass);
  free(auxCentroids);
  free(distCentroids);

  end = omp_get_wtime();
  printf("\nComputation: %f seconds\n", end - start);

  start = omp_get_wtime();
  end = omp_get_wtime();
  printf("\nMemory deallocation: %f seconds\n", end - start);

  return 0;
}

/* Helper functions */
void showFileError(int error, char *filename) {
  printf("Error\n");
  switch (error) {
  case -1:
    fprintf(stderr, "\tFile %s has too many columns.\n", filename);
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

int readInput(char *filename, int *lines, int *samples) {
  FILE *fp;
  char line[MAXLINE];
  char *ptr;
  const char *delim = "\t";
  int contlines = 0, contsamples = 0;

  if ((fp = fopen(filename, "r")) != NULL) {
    while (fgets(line, MAXLINE, fp) != NULL) {
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

int readInput2(char *filename, float *data) {
  FILE *fp;
  char line[MAXLINE];
  char *ptr;
  const char *delim = "\t";
  int i = 0;

  if ((fp = fopen(filename, "rt")) != NULL) {
    while (fgets(line, MAXLINE, fp) != NULL) {
      ptr = strtok(line, delim);
      while (ptr != NULL) {
        data[i++] = atof(ptr);
        ptr = strtok(NULL, delim);
      }
    }
    fclose(fp);
    return 0;
  } else {
    return -2;
  }
}

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

void initCentroids(const float *data, float *centroids, int *centroidPos,
                   int samples, int K) {
  for (int i = 0; i < K; i++) {
    memcpy(&centroids[i * samples], &data[centroidPos[i] * samples],
           samples * sizeof(float));
  }
}

float euclideanDistance(float *point, float *center, int samples) {
  float dist = 0.0;
  for (int i = 0; i < samples; i++) {
    dist += (point[i] - center[i]) * (point[i] - center[i]);
  }
  return sqrt(dist);
}

void zeroFloatMatrix(float *matrix, int rows, int columns) {
  memset(matrix, 0, rows * columns * sizeof(float));
}

void zeroIntArray(int *array, int size) {
  memset(array, 0, size * sizeof(int));
}
