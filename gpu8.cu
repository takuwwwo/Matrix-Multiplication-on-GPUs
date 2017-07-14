/*
N*N行列の行列積。共有メモリを用いる一般的な方法。
1ブロックが1列を担当。1スレッドは1要素を担当、スレッド数が足りなければその分ループ。

行列ABのBの計算する列を共有メモリに移して実行。
*/

#include "book.h"
#include <cublas_v2.h>
#include <curand.h>
#include <sys/time.h>

#define N 1024
#define Block 8192
#define Thread 1024
#define BLOCK_SIZE 32

void GPU_fill_rand(float *a, int row, int col){
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, a, row*col);
}

__global__ void transpose(float *a){                            //OK
  long tid = blockIdx.x * Thread + threadIdx.x;
  while(tid < N * N){
    int row = tid / N;
    int col = tid % N;
    if(row < col){
      float x = a[row*N+col];
      a[row*N+col] = a[col*N+row];
      a[col*N+row] = x;
    }
    tid += Thread * Block;
  }
}

__global__ void mmul3(float *a, float *b, float *c){
  int row = threadIdx.y, col = threadIdx.x;
  int arow = blockIdx.y*BLOCK_SIZE + row;
  int bcol = blockIdx.x*BLOCK_SIZE + col;

  __shared__ float cachea[BLOCK_SIZE][BLOCK_SIZE*2];
  __shared__ float cacheb[BLOCK_SIZE*2][BLOCK_SIZE];
  float sum = 0.0;

  int ax = arow*N+col;
  int bx = row*N+bcol;
  for(int i = 0; i < N; ){
    cachea[row][col] = a[ax+i];
    cacheb[row][col] = b[bx+i*N];
    i += BLOCK_SIZE;
    cachea[row][col+BLOCK_SIZE] = a[ax+i];
    cacheb[row+BLOCK_SIZE][col] = b[bx+i*N];
    i += BLOCK_SIZE;
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < 64; j++){
      sum += cachea[row][j]*cacheb[j][col];
    }
    __syncthreads();
  }
  c[arow*N+bcol] = sum;
}

__device__ void update2(float *a, float b, float *c){
  for(int i = 0; i < 16; i++) c[i] += a[i*4] * b;
}

__global__ void gpu8(float *a, float *b, float *c, int n){
  __shared__ float as[16*65];

  float cr[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int nDiv64 = n/64;
  int srow = threadIdx.y;
  int srow4 = srow*4;
  int scol = threadIdx.x;
  int tid = srow*16+scol;
  int aNext = (16*blockIdx.y+srow)*n+scol*4;
  int bNext = 128*blockIdx.x + tid;
  int cNext = 16*blockIdx.y*n + 128*blockIdx.x + tid;
  int nTimes2 = 2*n;
  int nTimes3 = 3*n;
  int nTimes4 = 4*n;

  a += aNext;
  b += bNext;
  c += cNext;

  for(int i = 0; i < nDiv64; i++){
    #pragma unroll
    for(int j = 0; j < 4; j++){
      as[scol][srow4+j] = a[j];
    }
    #pragma unroll
    for(int j = 0; j < 4; j++){
      as[scol][srow4+32+j] = a[nTimes2*4+j];
    }
    __syncthreads();


    float br0 = b[0];
    float br1 = b[n];
    float br2 = b[nTimes2];
    float br3 = b[nTimes3];

    b += nTimes4;

    #pragma unroll
    for(int k = 0; k < 15; k++){
      update2(&as[k][0], br0, cr); br0 = b[0];
      update2(&as[k][1], br1, cr); br1 = b[n];
      update2(&as[k][2], br2, cr); br2 = b[nTimes2];
      update2(&as[k][3], br3, cr); br3 = b[nTimes3];

      b += nTimes4;
    }
    update2(&as[15][0], br0, cr);
    update2(&as[15][1], br1, cr);
    update2(&as[15][2], br2, cr);
    update2(&as[15][3], br3, cr);

    a += 64;
    __syncthreads();
  }
  for(int j = 0; j < 16; j++){
    c[0] = cr[j];
    c += n;
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
  float *a, *b, *c1, *c2, *c3, *c4;
  float *dev_a, *dev_b, *dev_c, *dev_c2, *dev_c3;
  struct timeval t0, t1;

  a = (float *)malloc(N*N*sizeof(float));
  b = (float *)malloc(N*N*sizeof(float));
  c1 = (float *)malloc(N*N*sizeof(float));
  c2 = (float *)malloc(N*N*sizeof(float));
  c3 = (float *)malloc(N*N*sizeof(float));
  c4 = (float *)malloc(N*N*sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c2, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c3, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMemset(dev_c, 0, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMemset(dev_c2, 0, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMemset(dev_c3, 0, N*N*sizeof(float)));

  GPU_fill_rand(dev_a, N, N);
  GPU_fill_rand(dev_b, N, N);

  HANDLE_ERROR(cudaMemcpy(a, dev_a, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(b, dev_b, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  //mmulcpu(a, b, c4);

  gettimeofday(&t0, NULL);
  //mmul<<<Block, Thread>>>(dev_a, dev_b, dev_c);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(mmul)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));
  HANDLE_ERROR(cudaMemcpy(c1, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  gettimeofday(&t0, NULL);
  dim3 block2(16, 16);
  dim3 grid2(N/128, N/16);
  gpu8<<<grid2, block2>>>(dev_a, dev_b, dev_c, N);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(gpu8)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));
  HANDLE_ERROR(cudaMemcpy(c1, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  gettimeofday(&t0, NULL);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(N/BLOCK_SIZE, N/BLOCK_SIZE);
  mmul3<<<grid, block>>>(dev_a, dev_b, dev_c2);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(mmul2)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));
  HANDLE_ERROR(cudaMemcpy(c2, dev_c2, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0, beta = 0.0;
  //transpose<<<Block, Thread>>>(dev_a);
  //transpose<<<Block, Thread>>>(dev_b);

  gettimeofday(&t0, NULL);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_b, N, dev_a, N, &beta, dev_c3, N);
  //transpose<<<Block, Thread>>>(dev_c3);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(cublas)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));

  HANDLE_ERROR(cudaMemcpy(c3, dev_c3, N*N*sizeof(float), cudaMemcpyDeviceToHost));
  cublasDestroy(handle);

  transpose<<<Block, Thread>>>(dev_a);
  transpose<<<Block, Thread>>>(dev_b);

  int wrong1 = 0, wrong2 = 0, wrong3 = 0, wrong4 = 0;
  for(int i = 0; i < N; i++){
    for(int j = 0; j < N; j++){
      if(c1[i*N+j] != c2[i*N+j]){
        wrong1++;
        if(wrong1 < 20) printf("c1:%f, c2:%f\n", c1[(i)*N+j], c2[(i)*N+j]);
      }
      if(c1[i*N+j] != c3[i*N+j]) wrong2++;
      if(c1[i*N+j] != c4[i*N+j]) wrong4++;
      if(c2[i*N+j] != c3[i*N+j]){
        wrong3++;
        if(wrong3 < 20) printf("c2:%f, c3:%f\n", c1[i*N+j], c3[i*N+j]);
      }
    }
  }

  //printm(c, N, N);
  //printm(c2, N, N);
  //printm(c3, N, N);
  //printm(c4, N, N);
  for(int i = 0; i < 10; i++){
    for(int j = 0; j < 10; j++){
      //printf("%f %f\n", c1[i*N+j], c2[i*N+j]);
    }
  }
  printf("wrong1 : %d\n", wrong1);
  printf("wrong2 : %d\n", wrong2);
  printf("wrong3 : %d\n", wrong3);
  //printf("wrong4 : %d\n", wrong4)
  //printm(c2, N, N);

  free(a);
  free(b);
  free(c1);
  free(c2);
  free(c3);
  free(c4);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_c2));
  HANDLE_ERROR(cudaFree(dev_c3));


  return 0;
}
