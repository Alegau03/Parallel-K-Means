/*
 * k-Means clustering algorithm
 * CUDA version
 * 
 * Implementazione parallela utilizzando CUDA
 * Parallel computing (Degree in Computer Engineering)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

#define MAXLINE 2000

// Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* Funzioni di utilità */
void checkCudaError(const char *message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void showFileError(int error, char* filename) {
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

int readInput(char* filename, int *lines, int *samples) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;

    contlines = 0;

    if ((fp=fopen(filename,"r"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL) {
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
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;

    if ((fp=fopen(filename,"rt"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {         
            ptr = strtok(line, delim);
            while(ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    } else {
        return -2; //No file found
    }
}

int writeResult(int *classMap, int lines, const char* filename) {    
    FILE *fp;

    if ((fp=fopen(filename,"wt"))!=NULL) {
        for(int i=0; i<lines; i++) {
            fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  

        return 0;
    } else {
        return -3; //No file found
    }
}

void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
    int i;
    int idx;
    for(i=0; i<K; i++) {
        idx = centroidPos[i];
        memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
    }
}

__device__ float euclideanDistance(float *point, float *center, int samples) {
    float dist = 0.0;
    for (int i = 0; i < samples; i++) {
        float diff = point[i] - center[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__global__ void assignClusters(float *data, float *centroids, int *classMap, int lines, int samples, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < lines) {
        float minDist = FLT_MAX;
        int bestCluster = -1;
        for(int k = 0; k < K; k++) {
            float dist = euclideanDistance(&data[idx * samples], &centroids[k * samples], samples);
            if(dist < minDist) {
                minDist = dist;
                bestCluster = k;
            }
        }
        classMap[idx] = bestCluster;
    }
}

__global__ void recomputeCentroids(float *data, float *centroids, int *classMap, int *pointsPerClass, int lines, int samples, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        for (int d = 0; d < samples; d++) {
            centroids[idx * samples + d] = 0.0;
        }
        pointsPerClass[idx] = 0;

        for (int i = 0; i < lines; i++) {
            if (classMap[i] == idx) {
                pointsPerClass[idx]++;
                for (int d = 0; d < samples; d++) {
                    centroids[idx * samples + d] += data[i * samples + d];
                }
            }
        }

        if (pointsPerClass[idx] > 0) {
            for (int d = 0; d < samples; d++) {
                centroids[idx * samples + d] /= pointsPerClass[idx];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Validazione input
    if(argc != 7) {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
        exit(-1);
    }

    // Lettura dati
    int lines = 0, samples = 0;
    int error = readInput(argv[1], &lines, &samples);
    if(error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }

    float *data = (float*)calloc(lines * samples, sizeof(float));
    if(data == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }
    error = readInput2(argv[1], data);
    if(error != 0) {
        showFileError(error, argv[1]);
        exit(error);
    }

    // Parametri
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    int *centroidPos = (int*)calloc(K, sizeof(int));
    float *centroids = (float*)calloc(K * samples, sizeof(float));
    int *classMap = (int*)calloc(lines, sizeof(int));

    if(centroidPos == NULL || centroids == NULL || classMap == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(-4);
    }

    // Inizializzazione centroidi
    srand(0);
    for(int i = 0; i < K; i++) {
        centroidPos[i] = rand() % lines;
    }
    initCentroids(data, centroids, centroidPos, samples, K);

    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    // Allocazione memoria su GPU
    float *d_data, *d_centroids;
    int *d_classMap, *d_pointsPerClass;
    cudaMalloc(&d_data, lines * samples * sizeof(float));
    cudaMalloc(&d_centroids, K * samples * sizeof(float));
    cudaMalloc(&d_classMap, lines * sizeof(int));
    cudaMalloc(&d_pointsPerClass, K * sizeof(int));
    checkCudaError("Memory allocation on GPU");

    // Copia dei dati sulla GPU
    cudaMemcpy(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError("Data copy to GPU");

    int it = 0, changes = 0;
    do {
        it++;
        changes = 0;

        // Passo 1: Assegnazione dei punti ai centroidi
        assignClusters<<<(lines + 255) / 256, 256>>>(d_data, d_centroids, d_classMap, lines, samples, K);
        cudaDeviceSynchronize();
        checkCudaError("Cluster assignment kernel");

        // Passo 2: Ricalcolo dei centroidi
        recomputeCentroids<<<(K + 255) / 256, 256>>>(d_data, d_centroids, d_classMap, d_pointsPerClass, lines, samples, K);
        cudaDeviceSynchronize();
        checkCudaError("Centroid recomputation kernel");

        cudaMemcpy(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost);
        for(int i = 0; i < lines; i++) {
            if(classMap[i] != classMap[i]) {
                changes++;
            }
        }
    } while(changes > minChanges && it < maxIterations);

    // Copia risultati dalla GPU
    cudaMemcpy(centroids, d_centroids, K * samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Salvataggio risultati
    error = writeResult(classMap, lines, argv[6]);
    if(error != 0) {
        showFileError(error, argv[6]);
        exit(error);
    }

    // Libera memoria
    free(data);
    free(classMap);
    free(centroidPos);
    free(centroids);
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_classMap);
    cudaFree(d_pointsPerClass);

    return 0;
}