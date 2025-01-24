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

float euclideanDistance(float *point, float *center, int samples) {
  float dist = 0.0;
  for (int i = 0; i < samples; i++) {
    dist += (point[i] - center[i]) * (point[i] - center[i]);
  }
  return dist;
}


#define UNROLL 2

float euclideanDistance(float *point, float *center, int samples) {
    float dist = 0.0f;
    int blockSize = 32; // Dimensione del blocco ottimale per la cache L1
    int i, j;
    float diff;

    // Calcolo a blocchi
    for (i = 0; i < samples; i += blockSize) {
        float blockDist = 0.0f; // Accumulatore temporaneo per il blocco

        // Ciclo interno con unrolling
        for (j = i; j < i + blockSize && (j + UNROLL - 1) < samples; j += UNROLL) {
            // Calcola le differenze per UNROLL iterazioni
            diff = (point[j] - center[j]) * (point[j] - center[j]);
            diff += (point[j + 1] - center[j + 1]) * (point[j + 1] - center[j + 1]);
            blockDist += diff;
        }

        // Gestione degli elementi rimanenti
        for (; j < i + blockSize && j < samples; j++) {
            diff = (point[j] - center[j]) * (point[j] - center[j]);
            blockDist += diff;
        }

        dist += blockDist; // Somma il risultato del blocco
    }

    return dist; // Restituisce la distanza al quadrato
}
