/*
 * k-Means clustering algorithm
 *
 * MPI + OpenMP version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.1
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * Modified for hybrid MPI+OpenMP
 */
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAXLINE 2000
#define MAXCAD 200
#define UNROLL 2
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
    char line[MAXLINE] = "";
    const char *delim = "\t";
    int contlines = 0, contsamples = 0;

    if ((fp = fopen(filename, "r")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            contlines++;
            contsamples = 0;
            strtok(line, delim);
            while (strtok(NULL, delim) != NULL) contsamples++;
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
    char line[MAXLINE] = "";
    const char *delim = "\t";
    int i = 0;

    if ((fp = fopen(filename, "rt")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            char *ptr = strtok(line, delim);
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

void zeroFloatMatrix(float *matrix, int rows, int columns) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            matrix[i * columns + j] = 0.0f;
}

void zeroIntArray(int *array, int size) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < size; i++)
        array[i] = 0;
}

float euclideanDistance(float *point, float *center, int samples) {
    float dist = 0.0f;
    #pragma omp parallel for reduction(+ : dist) schedule(static)
    for (int i = 0; i < samples; i++) {
        float diff = point[i] - center[i];
        dist += diff * diff;
    }
    return dist;
}

void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K) {
    for (int i = 0; i < K; i++) {
        int idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], samples * sizeof(float));
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    double start, end;
    start = MPI_Wtime();

    if (argc != 7) {
        if (rank == 0) {
            fprintf(stderr, "EXECUTION ERROR: Invalid parameters.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int lines = 0, samples = 0;
    int error = readInput(argv[1], &lines, &samples);
    if (error != 0) {
        if (rank == 0) showFileError(error, argv[1]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    float *data = (float *)calloc(lines * samples, sizeof(float));
    if (data == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    error = readInput2(argv[1], data);
    if (error != 0) {
        if (rank == 0) showFileError(error, argv[1]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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

    srand(0);
    for (int i = 0; i < K; i++) centroidPos[i] = rand() % lines;
    initCentroids(data, centroids, centroidPos, samples, K);

    int localLines = lines / comm_size;
    int remainder = lines % comm_size;
    int startIdx = rank * localLines + (rank < remainder ? rank : remainder);
    int count = localLines + (rank < remainder);

    int *localClassMap = (int *)calloc(count, sizeof(int));
    float *localAuxCentroids = (float *)calloc(K * samples, sizeof(float));
    int *localPointsPerClass = (int *)calloc(K, sizeof(int));

    for (int it = 0; it < maxIterations; it++) {
        zeroFloatMatrix(localAuxCentroids, K, samples);
        zeroIntArray(localPointsPerClass, K);

        int changes = 0;  // Dichiarazione della variabile changes

        #pragma omp parallel for reduction(+ : changes) schedule(static)
        for (int i = 0; i < count; i++) {
            int globalIdx = startIdx + i;
            float minDist = FLT_MAX;
            int bestClass = 0;
            for (int j = 0; j < K; j++) {
                float dist = euclideanDistance(&data[globalIdx * samples], &centroids[j * samples], samples);
                if (dist < minDist) {
                    minDist = dist;
                    bestClass = j;
                }
            }
            if (localClassMap[i] != bestClass) {
                changes++;
            }
            localClassMap[i] = bestClass;
            #pragma omp atomic
            localPointsPerClass[bestClass]++;

            for (int j = 0; j < samples; j++) {
                #pragma omp atomic
                localAuxCentroids[bestClass * samples + j] += data[globalIdx * samples + j];
            }
        }

        int *globalPointsPerClass = (int *)calloc(K, sizeof(int));
        float *globalAuxCentroids = (float *)calloc(K * samples, sizeof(float));

        MPI_Allreduce(localPointsPerClass, globalPointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localAuxCentroids, globalAuxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < K; i++) {
            if (globalPointsPerClass[i] > 0) {
                for (int j = 0; j < samples; j++) {
                    globalAuxCentroids[i * samples + j] /= globalPointsPerClass[i];
                }
            }
        }

        memcpy(centroids, globalAuxCentroids, K * samples * sizeof(float));

        free(globalPointsPerClass);
        free(globalAuxCentroids);
    }

    free(data);
    free(centroidPos);
    free(centroids);
    free(classMap);
    free(localClassMap);
    free(localAuxCentroids);
    free(localPointsPerClass);

    MPI_Finalize();
    return 0;
}
