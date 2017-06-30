/*
N*N行列の行列積。共有メモリを用いない簡単な方法。
1スレッドが1要素の計算を担当。すなわち、1スレッドの計算量はN.
よって、必要なスレッド数はN*N.

各行を(2*Thread-1)/N個のブロックによって計算する。
*/

#include "book.h"
#include <cublas_v2.h>
#include <curand.h>

#define N 1
#define Block 8192
#define Thread 1024

void GPU_fill_rand(float *a, int row, int col) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, a, row*col);
}

void GPU_blas_mmul(const float *a, const float *b, float *c, const int m, const int k, const int n){
  int lda = m, ldb = k, ldc = m;
  float alf = 1.0;
  float bet = 0.0;
  float *alpha = &alf;
  float *beta = &bet;

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  cublasDestroy(handle);
}

__global__ void mmul(float *a, float *b, float *c){
  long tid = blockIdx.x * Thread + threadIdx.x;
  int row = tid / N;
  int col = tid % N;
  while(tid < N * N){
    c[tid] = 0.0;
    for(int i = 0; i < N; i++){
      c[tid] += a[row*N+i] * b[i*N+col];
    }
    tid += Thread * Block;
  }
}

void printm(float *a, int n, int m){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      printf("%f ", a[i*N+j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(void){
  float *a, *b, *c, *c2;
  float *dev_a, *dev_b, *dev_c, *dev_c2;

  a = (float *)malloc(N*N*sizeof(float));
  b = (float *)malloc(N*N*sizeof(float));
  c = (float *)malloc(N*N*sizeof(float));
  c2 = (float *)malloc(N*N*sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c2, N*N*sizeof(float)));

  GPU_fill_rand(dev_a, N, N);
  GPU_fill_rand(dev_b, N, N);

  HANDLE_ERROR(cudaMemcpy(a, dev_a, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(b, dev_b, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  //HANDLE_ERROR(cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  printm(a, N, N);
  printm(b, N, N);

  mmul<<<Block, Thread>>>(dev_a, dev_b, dev_c);
  GPU_blas_mmul(dev_a, dev_b, dev_c2, N, N, N);


  HANDLE_ERROR(cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(c2, dev_c2, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  printm(c, N, N);
  printm(c2, N, N);

  free(a);
  free(b);
  free(c);
  free(c2);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_c2));

  return 0;
}
