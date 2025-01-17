#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <mpi.h>
#include <omp.h>

#define MAXLINE 2000
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case -1:
            fprintf(stderr, "\tFile %s has too many columns.\n", filename);
            fprintf(stderr, "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
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

int readInput(char* filename, int *lines, int *samples) {
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

int readInput2(char* filename, float* data) {
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

int writeResult(int *classMap, int lines, const char* filename) {
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

void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
    for (int i = 0; i < K; i++) {
        memcpy(&centroids[i * samples], &data[centroidPos[i] * samples], samples * sizeof(float));
    }
}

static inline float distanceSquared(const float *point, const float *center, int samples) {
    float dist2 = 0.0f;
    for (int i = 0; i < samples; i++) {
        float diff = point[i] - center[i];
        dist2 += diff * diff;
    }
    return dist2;
}

void zeroFloatMatrix(float *matrix, int rows, int columns) {
    memset(matrix, 0, rows * columns * sizeof(float));
}

void zeroIntArray(int *array, int size) {
    memset(array, 0, size * sizeof(int));
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    double start, end;
    start = MPI_Wtime();

    if (argc != 7) {
        if (rank == 0) {
            fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
            fprintf(stderr, "Usage: mpirun -np <procs> ./KMEANS_MPI_OMP <InputFile> <K> <MaxIter> <Min%%Change> <Threshold> <OutputFile>\n");
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

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

    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    float *data = (float *)malloc(lines * samples * sizeof(float));
    if (!data) {
        fprintf(stderr, "Memory allocation error (data).\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        int error = readInput2(argv[1], data);
        if (error != 0) {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    MPI_Bcast(data, lines * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *centroids = (float *)calloc(K * samples, sizeof(float));
    int *centroidPos = NULL;
    if (rank == 0) {
        centroidPos = (int *)malloc(K * sizeof(int));
        if (!centroidPos) {
            fprintf(stderr, "Memory allocation error (centroidPos).\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        srand(0);
        for (int i = 0; i < K; i++) {
            centroidPos[i] = rand() % lines;
        }
        initCentroids(data, centroids, centroidPos, samples, K);
    }
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int localSize = lines / numProcs;
    int remainder = lines % numProcs;
    int startIndex = (rank < remainder) ? rank * (localSize + 1) : remainder * (localSize + 1) + (rank - remainder) * localSize;
    int endIndex = (rank < remainder) ? startIndex + (localSize + 1) : startIndex + localSize;
    int myNumPoints = endIndex - startIndex;

    int *classMapLocal = (int *)calloc(myNumPoints, sizeof(int));
    float *auxCentroidsLocal = (float *)calloc(K * samples, sizeof(float));
    int *pointsPerClassLocal = (int *)calloc(K, sizeof(int));

    int globalChanges;
    float maxDist;
    int continueFlag = 1;

    printf("\nStarting computation...\n");
    for (int it = 0; it < maxIterations && continueFlag; ++it) {
        double iter_start = MPI_Wtime();

        int localChanges = 0;
        zeroFloatMatrix(auxCentroidsLocal, K, samples);
        zeroIntArray(pointsPerClassLocal, K);

        #pragma omp parallel for reduction(+:localChanges) schedule(static)
        for (int i = 0; i < myNumPoints; i++) {
            int realIndex = startIndex + i;
            float minDist2 = FLT_MAX;
            int bestClass = -1;
            for (int j = 0; j < K; j++) {
                float dist2 = distanceSquared(&data[realIndex * samples], &centroids[j * samples], samples);
                if (dist2 < minDist2) {
                    minDist2 = dist2;
                    bestClass = j;
                }
            }
            if (classMapLocal[i] != bestClass + 1) {
                localChanges++;
                classMapLocal[i] = bestClass + 1;
            }
            for (int d = 0; d < samples; d++) {
                #pragma omp atomic
                auxCentroidsLocal[bestClass * samples + d] += data[realIndex * samples + d];
            }
            #pragma omp atomic
            pointsPerClassLocal[bestClass]++;
        }

        MPI_Allreduce(&localChanges, &globalChanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        float *globalAuxCentroids = (rank == 0) ? (float *)calloc(K * samples, sizeof(float)) : NULL;
        int *globalPointsPerClass = (rank == 0) ? (int *)calloc(K, sizeof(int)) : NULL;

        MPI_Reduce(auxCentroidsLocal, globalAuxCentroids, K * samples, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(pointsPerClassLocal, globalPointsPerClass, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            maxDist = 0.0f;
            for (int j = 0; j < K; j++) {
                if (globalPointsPerClass[j] > 0) {
                    for (int d = 0; d < samples; d++) {
                        globalAuxCentroids[j * samples + d] /= globalPointsPerClass[j];
                    }
                }
                maxDist = MAX(maxDist, sqrt(distanceSquared(&centroids[j * samples], &globalAuxCentroids[j * samples], samples)));
            }
            memcpy(centroids, globalAuxCentroids, K * samples * sizeof(float));

            printf("[%d] Cluster changes: %d\tMax. centroid distance: %f\n", it + 1, globalChanges, maxDist);

            free(globalAuxCentroids);
            free(globalPointsPerClass);
        }

        MPI_Bcast(&maxDist, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

        continueFlag = (globalChanges > minChanges) && (maxDist > maxThreshold);
        MPI_Bcast(&continueFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        double iter_end = MPI_Wtime();
        if (rank == 0) {
            printf("Iteration %d time: %f seconds\n", it + 1, iter_end - iter_start);
        }
    }

    if (rank == 0) {
        int *globalClassMap = (int *)calloc(lines, sizeof(int));
        MPI_Gather(classMapLocal, myNumPoints, MPI_INT, globalClassMap, myNumPoints, MPI_INT, 0, MPI_COMM_WORLD);
        writeResult(globalClassMap, lines, argv[6]);
        free(globalClassMap);

        end = MPI_Wtime();
        printf("\nComputation: %f seconds\n", end - start);
    } else {
        MPI_Gather(classMapLocal, myNumPoints, MPI_INT, NULL, myNumPoints, MPI_INT, 0, MPI_COMM_WORLD);
    }

    free(data);
    free(centroids);
    free(classMapLocal);
    free(auxCentroidsLocal);
    free(pointsPerClassLocal);
    if (rank == 0) free(centroidPos);

    MPI_Finalize();
    return 0;
}
