#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> // for AVX intrinsics

// Define matrix dimension; must be a multiple of 8.
#ifndef N
#define N 256
#endif

// Scalar implementation using 2D array indexing.
void scalar_2Dimplementation(float A[N][N], float B[N][N], float C[N][N]) {
    // For each element, use the transposed value from A (i.e. A[j][i]) and multiply with B[i][j]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[j][i] * B[i][j];
        }
    }
}

// Scalar implementation using 1D pointer arithmetic style.
// (Note: Since our matrices are declared as 2D arrays, we use the same logic.)
void scalar_1Dimplementation(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // Treat A as if it were stored in 1D in row-major order.
            // For the transposed element, A[j][i] is at index j*N + i.
            C[i][j] = A[j][i] * B[i][j];
        }
    }
}

// SIMD implementation using AVX intrinsics.
// We perform the transposition in 8x8 blocks and then do element-wise multiplication.
void simd_implementation(float A[N][N], float B[N][N], float C[N][N]) {
    // Process the matrix in 8x8 blocks.
    for (int i0 = 0; i0 < N; i0 += 8) {
        for (int j0 = 0; j0 < N; j0 += 8) {
            // Load an 8x8 block from A.
            // Since we need the transposed block (A^T), we load rows from A with row index (j0+k)
            __m256 row0 = _mm256_loadu_ps(&A[j0 + 0][i0]);
            __m256 row1 = _mm256_loadu_ps(&A[j0 + 1][i0]);
            __m256 row2 = _mm256_loadu_ps(&A[j0 + 2][i0]);
            __m256 row3 = _mm256_loadu_ps(&A[j0 + 3][i0]);
            __m256 row4 = _mm256_loadu_ps(&A[j0 + 4][i0]);
            __m256 row5 = _mm256_loadu_ps(&A[j0 + 5][i0]);
            __m256 row6 = _mm256_loadu_ps(&A[j0 + 6][i0]);
            __m256 row7 = _mm256_loadu_ps(&A[j0 + 7][i0]);

            // First step: unpack the rows pairwise.
            __m256 t0 = _mm256_unpacklo_ps(row0, row1);
            __m256 t1 = _mm256_unpackhi_ps(row0, row1);
            __m256 t2 = _mm256_unpacklo_ps(row2, row3);
            __m256 t3 = _mm256_unpackhi_ps(row2, row3);
            __m256 t4 = _mm256_unpacklo_ps(row4, row5);
            __m256 t5 = _mm256_unpackhi_ps(row4, row5);
            __m256 t6 = _mm256_unpacklo_ps(row6, row7);
            __m256 t7 = _mm256_unpackhi_ps(row6, row7);

            // Second step: shuffle to interleave.
            __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
            __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
            __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
            __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
            __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
            __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
            __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
            __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

            // Third step: permute to combine 128-bit lanes.
            __m256 col0 = _mm256_permute2f128_ps(s0, s4, 0x20);
            __m256 col1 = _mm256_permute2f128_ps(s1, s5, 0x20);
            __m256 col2 = _mm256_permute2f128_ps(s2, s6, 0x20);
            __m256 col3 = _mm256_permute2f128_ps(s3, s7, 0x20);
            __m256 col4 = _mm256_permute2f128_ps(s0, s4, 0x31);
            __m256 col5 = _mm256_permute2f128_ps(s1, s5, 0x31);
            __m256 col6 = _mm256_permute2f128_ps(s2, s6, 0x31);
            __m256 col7 = _mm256_permute2f128_ps(s3, s7, 0x31);

            // At this point, col0 to col7 contain the transposed 8×8 block.
            // They correspond to rows of A^T; that is, col0 holds the elements that should go to row (i0+0)
            // of the result (remember C[i][j] = A^T[i][j] * B[i][j]).

            // Load the corresponding 8x8 block from B.
            __m256 b0 = _mm256_loadu_ps(&B[i0 + 0][j0]);
            __m256 b1 = _mm256_loadu_ps(&B[i0 + 1][j0]);
            __m256 b2 = _mm256_loadu_ps(&B[i0 + 2][j0]);
            __m256 b3 = _mm256_loadu_ps(&B[i0 + 3][j0]);
            __m256 b4 = _mm256_loadu_ps(&B[i0 + 4][j0]);
            __m256 b5 = _mm256_loadu_ps(&B[i0 + 5][j0]);
            __m256 b6 = _mm256_loadu_ps(&B[i0 + 6][j0]);
            __m256 b7 = _mm256_loadu_ps(&B[i0 + 7][j0]);

            // Perform element-wise multiplication.
            __m256 res0 = _mm256_mul_ps(col0, b0);
            __m256 res1 = _mm256_mul_ps(col1, b1);
            __m256 res2 = _mm256_mul_ps(col2, b2);
            __m256 res3 = _mm256_mul_ps(col3, b3);
            __m256 res4 = _mm256_mul_ps(col4, b4);
            __m256 res5 = _mm256_mul_ps(col5, b5);
            __m256 res6 = _mm256_mul_ps(col6, b6);
            __m256 res7 = _mm256_mul_ps(col7, b7);

            // Store the results into C.
            _mm256_storeu_ps(&C[i0 + 0][j0], res0);
            _mm256_storeu_ps(&C[i0 + 1][j0], res1);
            _mm256_storeu_ps(&C[i0 + 2][j0], res2);
            _mm256_storeu_ps(&C[i0 + 3][j0], res3);
            _mm256_storeu_ps(&C[i0 + 4][j0], res4);
            _mm256_storeu_ps(&C[i0 + 5][j0], res5);
            _mm256_storeu_ps(&C[i0 + 6][j0], res6);
            _mm256_storeu_ps(&C[i0 + 7][j0], res7);
        }
    }
}

int main(int argc, char* argv[]) {
    /* 
       For simplicity, this code uses a fixed compile-time matrix size N.
       If a command-line argument is provided, we check that it matches N and is a multiple of 8.
    */
    int matrix_size = N;
    if (argc > 1) {
        matrix_size = atoi(argv[1]);
        if (matrix_size != N || (matrix_size % 8) != 0) {
            printf("Error: Matrix size must be %d (a multiple of 8).\n", N);
            return -1;
        }
    }
    
    // Allocate matrices as 2D arrays (non-contiguous allocation)
    float **A = (float**) malloc(matrix_size * sizeof(float*));
    float **B = (float**) malloc(matrix_size * sizeof(float*));
    float **C = (float**) malloc(matrix_size * sizeof(float*));
    for (int i = 0; i < matrix_size; i++) {
        A[i] = (float*) malloc(matrix_size * sizeof(float));
        B[i] = (float*) malloc(matrix_size * sizeof(float));
        C[i] = (float*) malloc(matrix_size * sizeof(float));
    }
    
    // Initialize matrices A and B with random values.
    srand((unsigned int)time(NULL));
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A[i][j] = (float)rand() / RAND_MAX;
            B[i][j] = (float)rand() / RAND_MAX;
        }
    }
    
    clock_t start, end;
    double cpu_time_used;
    
    // ---------------------------
    // Scalar 2D Implementation
    // ---------------------------
    start = clock();
    scalar_2Dimplementation((float (*)[N])A, (float (*)[N])B, (float (*)[N])C);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Scalar 2D time: %f seconds\n", cpu_time_used);
    
    // ---------------------------
    // Scalar 1D Implementation
    // ---------------------------
    start = clock();
    scalar_1Dimplementation((float (*)[N])A, (float (*)[N])B, (float (*)[N])C);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Scalar 1D time: %f seconds\n", cpu_time_used);
    
    // ---------------------------
    // SIMD Implementation
    // ---------------------------
    // For the SIMD version, we need a contiguous block of memory.
    float *A_contig = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    float *B_contig = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    float *C_contig = (float*) malloc(matrix_size * matrix_size * sizeof(float));
    
    // Copy data from A and B into contiguous arrays (row-major order).
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            A_contig[i * matrix_size + j] = A[i][j];
            B_contig[i * matrix_size + j] = B[i][j];
        }
    }
    
    // Reinterpret the contiguous memory as 2D arrays.
    float (*A_simd)[N] = (float (*)[N]) A_contig;
    float (*B_simd)[N] = (float (*)[N]) B_contig;
    float (*C_simd)[N] = (float (*)[N]) C_contig;
    
    start = clock();
    simd_implementation(A_simd, B_simd, C_simd);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("SIMD time: %f seconds\n", cpu_time_used);
    
    // Optionally, you could verify correctness here by comparing C and C_contig.
    
    // Free allocated memory.
    for (int i = 0; i < matrix_size; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);
    free(A_contig);
    free(B_contig);
    free(C_contig);
    
    return 0;
}
