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

__global__ void mmul(int *a, int *b, int *c){
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

void mmulcpu(int *a, int *b, int *c){
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      for(int k = 0; k < N; k++){
        c[i*N+j] += a[i*N+k]*b[k*N+j];
      }
    }
  }
}

int main(void){
  int *a, *b, *c;
  int *dev_a, *dev_b, *dev_c;

  a = (int *)malloc(N*N*sizeof(int));
  b = (int *)malloc(N*N*sizeof(int));
  c = (int *)malloc(N*N*sizeof(int));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*N*sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*N*sizeof(int)));

  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      a[i*N+j] = i - j;
      b[i*N+j] = i + j;
      c[i*N+j] = 0;
    }
  }

  HANDLE_ERROR(cudaMemcpy(dev_a, a, N*N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_b, b, N*N*sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_c, c, N*N*sizeof(int), cudaMemcpyHostToDevice));

  mmul<<<Block, Thread>>>(dev_a, dev_b, dev_c);

  HANDLE_ERROR(cudaMemcpy(c, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost));

  /*
  int  *a2, *b2, *c2;
  a2 = (int *)malloc(N*N*sizeof(int));
  b2 = (int *)malloc(N*N*sizeof(int));
  c2 = (int *)malloc(N*N*sizeof(int));
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
