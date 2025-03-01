#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>  // For AVX intrinsics

// Default matrix dimension (must be a multiple of 8)

// Macro to index a 2D matrix stored in row-major order in a 1D array.
#define IDX(i, j, n) ((i) * (n) + (j))

// Scalar implementation using 2D-like indexing.
void scalar_2Dimplementation(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Note: transposition means we use A[j][i] for the element at (i,j)
            C[IDX(i, j, n)] = A[IDX(j, i, n)] * B[IDX(i, j, n)];
        }
    }
}

// Scalar implementation using 1D indexing (conceptually similar).
void scalar_1Dimplementation(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = A[j*n + i] * B[i*n + j];
        }
    }
}

// SIMD implementation using AVX intrinsics.
// This version processes the matrices in 8x8 blocks.
// It assumes that 'n' is a multiple of 8.
void simd_implementation(const float *A, const float *B, float *C, int n) {
    // We will cast our contiguous memory blocks to a 2D array pointer
    // (relying on C99 variable-length arrays).
    float (*A2D)[n] = (float (*)[n]) A;
    float (*B2D)[n] = (float (*)[n]) B;
    float (*C2D)[n] = (float (*)[n]) C;

    for (int i0 = 0; i0 < n; i0 += 8) {
        for (int j0 = 0; j0 < n; j0 += 8) {
            // Load an 8x8 block from A.
            // We want the transposed block, so we load rows corresponding to columns.
            __m256 row0 = _mm256_loadu_ps(&A2D[j0 + 0][i0]);
            __m256 row1 = _mm256_loadu_ps(&A2D[j0 + 1][i0]);
            __m256 row2 = _mm256_loadu_ps(&A2D[j0 + 2][i0]);
            __m256 row3 = _mm256_loadu_ps(&A2D[j0 + 3][i0]);
            __m256 row4 = _mm256_loadu_ps(&A2D[j0 + 4][i0]);
            __m256 row5 = _mm256_loadu_ps(&A2D[j0 + 5][i0]);
            __m256 row6 = _mm256_loadu_ps(&A2D[j0 + 6][i0]);
            __m256 row7 = _mm256_loadu_ps(&A2D[j0 + 7][i0]);

            // Unpack and shuffle to perform an 8x8 matrix transposition.
            __m256 t0 = _mm256_unpacklo_ps(row0, row1);
            __m256 t1 = _mm256_unpackhi_ps(row0, row1);
            __m256 t2 = _mm256_unpacklo_ps(row2, row3);
            __m256 t3 = _mm256_unpackhi_ps(row2, row3);
            __m256 t4 = _mm256_unpacklo_ps(row4, row5);
            __m256 t5 = _mm256_unpackhi_ps(row4, row5);
            __m256 t6 = _mm256_unpacklo_ps(row6, row7);
            __m256 t7 = _mm256_unpackhi_ps(row6, row7);

            __m256 s0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1,0,1,0));
            __m256 s1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3,2,3,2));
            __m256 s2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1,0,1,0));
            __m256 s3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3,2,3,2));
            __m256 s4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1,0,1,0));
            __m256 s5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3,2,3,2));
            __m256 s6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1,0,1,0));
            __m256 s7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3,2,3,2));

            __m256 col0 = _mm256_permute2f128_ps(s0, s4, 0x20);
            __m256 col1 = _mm256_permute2f128_ps(s1, s5, 0x20);
            __m256 col2 = _mm256_permute2f128_ps(s2, s6, 0x20);
            __m256 col3 = _mm256_permute2f128_ps(s3, s7, 0x20);
            __m256 col4 = _mm256_permute2f128_ps(s0, s4, 0x31);
            __m256 col5 = _mm256_permute2f128_ps(s1, s5, 0x31);
            __m256 col6 = _mm256_permute2f128_ps(s2, s6, 0x31);
            __m256 col7 = _mm256_permute2f128_ps(s3, s7, 0x31);

            // Now, col0 ... col7 hold the transposed block from A.
            // Next, load the corresponding 8x8 block from B.
            __m256 b0 = _mm256_loadu_ps(&B2D[i0 + 0][j0]);
            __m256 b1 = _mm256_loadu_ps(&B2D[i0 + 1][j0]);
            __m256 b2 = _mm256_loadu_ps(&B2D[i0 + 2][j0]);
            __m256 b3 = _mm256_loadu_ps(&B2D[i0 + 3][j0]);
            __m256 b4 = _mm256_loadu_ps(&B2D[i0 + 4][j0]);
            __m256 b5 = _mm256_loadu_ps(&B2D[i0 + 5][j0]);
            __m256 b6 = _mm256_loadu_ps(&B2D[i0 + 6][j0]);
            __m256 b7 = _mm256_loadu_ps(&B2D[i0 + 7][j0]);

            // Element-wise multiplication.
            __m256 res0 = _mm256_mul_ps(col0, b0);
            __m256 res1 = _mm256_mul_ps(col1, b1);
            __m256 res2 = _mm256_mul_ps(col2, b2);
            __m256 res3 = _mm256_mul_ps(col3, b3);
            __m256 res4 = _mm256_mul_ps(col4, b4);
            __m256 res5 = _mm256_mul_ps(col5, b5);
            __m256 res6 = _mm256_mul_ps(col6, b6);
            __m256 res7 = _mm256_mul_ps(col7, b7);

            // Store the resulting block into C.
            _mm256_storeu_ps(&C2D[i0 + 0][j0], res0);
            _mm256_storeu_ps(&C2D[i0 + 1][j0], res1);
            _mm256_storeu_ps(&C2D[i0 + 2][j0], res2);
            _mm256_storeu_ps(&C2D[i0 + 3][j0], res3);
            _mm256_storeu_ps(&C2D[i0 + 4][j0], res4);
            _mm256_storeu_ps(&C2D[i0 + 5][j0], res5);
            _mm256_storeu_ps(&C2D[i0 + 6][j0], res6);
            _mm256_storeu_ps(&C2D[i0 + 7][j0], res7);
        }
    }
}

// Function to compare two matrices and return the maximum absolute difference.
float verify_results(const float *C_ref, const float *C_test, int n) {
    float max_diff = 0.0f;
    for (int i = 0; i < n * n; i++) {
        float diff = fabsf(C_ref[i] - C_test[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    return max_diff;
}

int main(int argc, char* argv[]) {
    int n = argc == 2 ? atoi(argv[1]) : 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0 || (n % 8) != 0) {
            printf("Error: Matrix size must be a positive multiple of 8.\n");
            return -1;
        }
    }
    printf("Matrix size: %d x %d\n", n, n);

    // Allocate contiguous memory for matrices A, B and results from each implementation.
    size_t bytes = n * n * sizeof(float);
    float *A       = (float *) malloc(bytes);
    float *B       = (float *) malloc(bytes);
    float *C_2D    = (float *) malloc(bytes);
    float *C_1D    = (float *) malloc(bytes);
    float *C_simd  = (float *) malloc(bytes);
    if (!A || !B || !C_2D || !C_1D || !C_simd) {
        printf("Memory allocation error\n");
        return -1;
    }

    // Initialize matrices A and B with random float values.
    srand((unsigned int) time(NULL));
    for (int i = 0; i < n * n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t start, end;
    double time_2D, time_1D, time_simd;

    // ---------------------------
    // Scalar 2D Implementation
    // ---------------------------
    start = clock();
    scalar_2Dimplementation(A, B, C_2D, n);
    end = clock();
    time_2D = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Scalar 2D time: %f seconds\n", time_2D);

    // ---------------------------
    // Scalar 1D Implementation
    // ---------------------------
    start = clock();
    scalar_1Dimplementation(A, B, C_1D, n);
    end = clock();
    time_1D = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Scalar 1D time: %f seconds\n", time_1D);

    // ---------------------------
    // SIMD Implementation
    // ---------------------------
    start = clock();
    simd_implementation(A, B, C_simd, n);
    end = clock();
    time_simd = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("SIMD time: %f seconds\n", time_simd);

    // ---------------------------
    // Speedup Calculation
    // ---------------------------
    // Here, we compute the speedup of the SIMD implementation relative to the scalar implementations.
    double speedup_2D = time_2D / time_simd;
    double speedup_1D = time_1D / time_simd;
    printf("Speedup (Scalar 2D / SIMD): %f\n", speedup_2D);
    printf("Speedup (Scalar 1D / SIMD): %f\n", speedup_1D);

    // ---------------------------
    // Verification: Compare SIMD result with scalar 2D result.
    // ---------------------------
    float max_diff = verify_results(C_2D, C_simd, n);
    if (max_diff > 1e-6f) {
        printf("Verification FAILED! Maximum difference: %e\n", max_diff);
    } else {
        printf("Verification PASSED! Maximum difference: %e\n", max_diff);
    }

    // Free allocated memory.
    free(A);
    free(B);
    free(C_2D);
    free(C_1D);
    free(C_simd);

    return 0;
}
