/*
 * k-Means clustering algorithm
 *
 * MPI version
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
#include <time.h>

#define MAXLINE 2000
#define MAXCAD 200

#define DISTRIBUTION(size, points, dist)                                       \
  int tmp = 0;                                                                 \
  while (points > 0) {                                                         \
    dist[(size + tmp) % size] += 1;                                            \
    tmp++;                                                                     \
    points--;                                                                  \
  }
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
#define BLOCKTIME(block)                                                       \
  {                                                                            \
    double start = MPI_Wtime();                                                \
    block double end = MPI_Wtime();                                            \
    printf("\n%s time : %lf\n ", #block, end - start);                         \
  }

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void sum_op(void *inP, void *inoutP, int *len, MPI_Datatype *dptr) {
  float *in = (float *)inP, *inout = (float *)inoutP;
  for (int i = 0; i < *len; i++) {
    inout[i] += in[i];
  }
}
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

Function initCentroids: This function copies the values of the initial
centroids, using their position in the input data structure as a reference
map.
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
float euclideanDistance(float *point, float *center, int samples) {
  float dist = 0.0;
  for (int i = 0; i < samples; i++) {
    dist += (point[i] - center[i]) * (point[i] - center[i]);
  }
  return sqrt(dist);
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
  /* 0. Initialize MPI */
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

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

  // pointPerClass: number of points classified in each class
  // auxCentroids: mean of the points in each class
  int *pointsPerClass = (int *)malloc(K * sizeof(int));
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

  int comm_size, *point_distribution, *offset, my_iteration, my_offset;
  int tmp_lines = lines;

  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  if (rank == (comm_size + 1) % comm_size) {
    point_distribution = calloc(comm_size, sizeof(int));
    offset = calloc(comm_size, sizeof(int));

    DISTRIBUTION(comm_size, tmp_lines, point_distribution);
    OFFSET(offset, point_distribution, comm_size);
    my_offset = offset[rank];
    my_iteration = point_distribution[rank];
  } else {
    int remainder = lines % comm_size;
    my_iteration = (rank < remainder) ? ((lines - remainder) / comm_size) + 1
                                      : ((lines - remainder) / comm_size);
    my_offset = (rank < remainder) ? rank * my_iteration
                                   : (remainder * (my_iteration + 1)) +
                                         ((rank - remainder) * my_iteration);
  }
  int *localClassMap = calloc(my_iteration, sizeof(int));
  int glob_changes;
  float *glob_auxCentroids;
  int *glob_pointsPerClass;
  int packet_size = (K * samples * sizeof(float)) + sizeof(int);
  void *packet;

  do {
    it++;

    // 1. Calculate the distance from each point to the centroid
    // Assign each point to the nearest centroid.
    changes = 0;
    for (int i = my_offset, it_2 = 0; i < my_offset + my_iteration; i++) {
      class = 1;
      minDist = FLT_MAX;
      for (j = 0; j < K; j++) {
        dist = euclideanDistance(&data[i * samples], &centroids[j * samples],
                                 samples);

        if (dist < minDist) {
          minDist = dist;
          class = j + 1;
        }
      }
      if (localClassMap[it_2] != class) {
        changes++;
      }
      localClassMap[it_2++] = class;
    }
    MPI_Reduce(&changes, &glob_changes, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    zeroIntArray(pointsPerClass, K);
    zeroFloatMatriz(auxCentroids, K, samples);
    for (int i = my_offset, it_2 = 0; i < my_offset + my_iteration; i++) {
      class = localClassMap[it_2++];
      pointsPerClass[class - 1] = pointsPerClass[class - 1] + 1;
      for (j = 0; j < samples; j++) {
        auxCentroids[(class - 1) * samples + j] += data[i * samples + j];
      }
    }

    if (rank == 0) {
      glob_auxCentroids = calloc(K * samples, sizeof(float));
      glob_pointsPerClass = calloc(K, sizeof(int));
    }
    packet = malloc(packet_size);

    MPI_Reduce(auxCentroids, glob_auxCentroids, K * samples, MPI_FLOAT, MPI_SUM,
               0, MPI_COMM_WORLD);

    MPI_Reduce(pointsPerClass, glob_pointsPerClass, K, MPI_INT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      for (i = 0; i < K; i++) {
        for (j = 0; j < samples; j++) {
          glob_auxCentroids[i * samples + j] /= glob_pointsPerClass[i];
        }
      }
      int position = 0;
      memcpy(centroids, glob_auxCentroids, (K * samples * sizeof(float)));
      MPI_Pack(centroids, K * samples, MPI_FLOAT, packet, packet_size,
               &position, MPI_COMM_WORLD);
      MPI_Pack(&glob_changes, 1, MPI_INT, packet, packet_size, &position,
               MPI_COMM_WORLD);
    }
    MPI_Bcast(packet, packet_size, MPI_PACKED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      changes = glob_changes;
      free(packet);
    } else {
      int position = 0;
      MPI_Unpack(packet, packet_size, &position, centroids, K * samples,
                 MPI_FLOAT, MPI_COMM_WORLD);
      MPI_Unpack(packet, packet_size, &position, &changes, 1, MPI_INT,
                 MPI_COMM_WORLD);
      free(packet);
    }
    maxDist = FLT_MIN;
    for (i = 0; i < K; i++) {
      distCentroids[i] = euclideanDistance(&centroids[i * samples],
                                           &auxCentroids[i * samples], samples);
      maxDist = MAX(maxDist, distCentroids[i]);
    }
    sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it,
            changes, maxDist);
    outputMsg = strcat(outputMsg, line);

  } while ((changes > minChanges) && (it < maxIterations) &&
           (maxDist > maxThreshold));

  MPI_Gatherv(localClassMap, my_iteration, MPI_INT, classMap,
              point_distribution, offset, MPI_INT, (comm_size + 1) % comm_size,
              MPI_COMM_WORLD);
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
             MPI_COMM_WORLD);
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
  if (rank == (comm_size + 1) % comm_size) {
    // Writing the classification of each point to the output file.
    error = writeResult(classMap, lines, argv[6]);
    if (error != 0) {
      showFileError(error, argv[6]);
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  // Free memory
  free(data);
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
