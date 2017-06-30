/*
N*N行列の行列積。共有メモリを用いない簡単な方法。
1スレッドが1要素の計算を担当。すなわち、1スレッドの計算量はN.
よって、必要なスレッド数はN*N.

各行を(2*Thread-1)/N個のブロックによって計算する。
*/

#include "book.h"

#define N 10000
#define Block 8192
#define Thread 1024

__global__ void mmul(float *a, float *b, float *c){
  long tid = blockIdx.x * Thread + threadIdx.x;
  int row = tid / N;
  int col = tid % N;
  while(tid < N * N){
    for(int i = 0; i < N; i++){
      c[tid] += a[row*N+i] * b[i*N+col];
    }
    tid += Thread * Block;
  }
}

void mmulcpu(float *a, float *b, float *c){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      for(int k = 0; k < N; k++){
        c[i*N+j] += a[i*N+k]*b[k*N+j];
      }
    }
  }
}

void printm(float *a, int n, int m){
  for(int i = 0; i < n; i++){
    for(int j = 0; j < m; j++){
      printf("%f ", a[i*N+j]);
    }
    printf("\n");
  }
}

int main(void){
  float *a, *b, *c;
  float *dev_a, *dev_b, *dev_c;

  a = (float *)malloc(N*N*sizeof(float));
  b = (float *)malloc(N*N*sizeof(float));
  c = (float *)malloc(N*N*sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*N*sizeof(float)));

  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      a[i*N+j] = i - j;
      b[i*N+j] = i + j;
      c[i*N+j] = 0;
    }
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*N*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*N*sizeof(float), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_c, c, N*N*sizeof(float), cudaMemcpyHostToDevice));

  dim3 grid(Block, Block);
  mmul<<<grid, Thread>>>(dev_a, dev_b, dev_c);

  HANDLE_ERROR(cudaMemcpy(c, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  /*
  float  *a2, *b2, *c2;
  a2 = (float *)malloc(N*N*sizeof(float));
  b2 = (float *)malloc(N*N*sizeof(float));
  c2 = (float *)malloc(N*N*sizeof(float));
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      a2[i*N+j] = i - j;
      b2[i*N+j] = i + j;
      c2[i*N+j] = 0;
    }
  }
  mmulcpu(a2, b2, c2);
  int wrong = 0;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if(c[i*N+j] != c2[i*N+j]) wrong++;
    }
  }
  printf("wrong : %d\n", wrong);
  free(a2);
  free(b2);
  free(c2);
  */

  free(a);
  free(b);
  free(c);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));

  return 0;
}
