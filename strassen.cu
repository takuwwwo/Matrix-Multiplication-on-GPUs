/*
mmul2 is the naive Matrix Multiplication on a GPU.
mmul is fast Matrix Multiplication on GPU by using Shared Memory.
strassen is Matrix Multiplication by Strassen algorithm.
The main program is the comparison of time between mmul, strassen and cublas.
*/

#include "book.h"
#include <cublas_v2.h>
#include <curand.h>
#include <sys/time.h>

#define N 16384
#define Block 8192
#define Thread 1024
#define BLOCK_SIZE 32
#define Share 4
#define tu1 4096

void GPU_fill_rand(float *a, int row, int col){
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  curandGenerateUniform(prng, a, row*col);
}

__global__ void mmul2(float *a, float *b, float *c){
  long tid = blockIdx.x * Thread + threadIdx.x;
  while(tid < N * N){
    int row = tid / N;
    int col = tid % N;
    c[tid] = 0;
    for(int i = 0; i < N; i++){
      c[row*N+col] += a[row*N+i] * b[i*N+col];
    }
    tid += Thread * Block;
  }
}

__global__ void add(float *a, float *b, float *c){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c[tid] = a[tid] + b[tid];
}

__global__ void sub(float *a, float *b, float *c){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c[tid] = a[tid] - b[tid];
}

__global__ void sub_add(float *a1, float *b1, float *c1, float *a2, float *b2, float *c2){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c1[tid] = a1[tid] - b1[tid];
  c2[tid] = a2[tid] + b2[tid];
}

__global__ void add_add(float *a1, float *b1, float *c1, float *a2, float *b2, float *c2){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c1[tid] = a1[tid] + b1[tid];
  c2[tid] = a2[tid] + b2[tid];
}

__global__ void wadd(float *a, float *c1, float *c2){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c1[tid] += a[tid];
  c2[tid] += a[tid];
}

__global__ void wsub_add(float *a, float *c1, float *c2){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  c1[tid] -= a[tid];
  c2[tid] += a[tid];
}

__global__ void madd(float *a, float *b){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  b[tid] += a[tid];
}

__global__ void msub(float *a, float *b){
  int tid = blockIdx.y*N*BLOCK_SIZE + blockIdx.x*BLOCK_SIZE + threadIdx.y*N + threadIdx.x;
  b[tid] -= a[tid];
}

__global__ void mmul(float *a, float *b, float *c, int n){
  int row = threadIdx.y, col = threadIdx.x;
  int arow = blockIdx.y*BLOCK_SIZE + row;
  int bcol = blockIdx.x*BLOCK_SIZE + col;

  __shared__ float cachea[BLOCK_SIZE][BLOCK_SIZE*Share];
  __shared__ float cacheb[BLOCK_SIZE*Share][BLOCK_SIZE];
  float sum = 0.0;

  int ax = arow*N+col;
  int bx = row*N+bcol;
  int bs = BLOCK_SIZE*Share;
  for(int i = 0; i < n; ){
    #pragma unroll
    for(int j = 0; j < Share; j++){
      cachea[row][col+j*BLOCK_SIZE] = a[ax+i];
      cacheb[row+j*BLOCK_SIZE][col] = b[bx+i*N];
      i += BLOCK_SIZE;
    }
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < bs; k++){
      sum += cachea[row][k]*cacheb[k][col];
    }
    __syncthreads();
  }
  c[arow*N+bcol] = sum;
}

void strassen(float *a, float *b, float *c, float *t, int n, cublasHandle_t handle, float* alpha, float* beta, cudaStream_t s0, cudaStream_t s1){
  if(n <= tu1){  //OK
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, alpha, b, N, a, N, beta, c, N);
  }
  else{
    int halfn = n/2;
    int p11 = 0, p12 = halfn, p21 = N*halfn, p22 = N*halfn + halfn;
    float *t1 = t;
    float *t2 = t+halfn*N;
    float *next = t + halfn;
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(halfn/BLOCK_SIZE, halfn/BLOCK_SIZE);

    sub<<<grid, block, 0, s0>>>(&a[p21], &a[p11], &c[p12]);
    add<<<grid, block, 0, s1>>>(&b[p11], &b[p12], &c[p21]);
    cudaThreadSynchronize();
    strassen(&c[p12], &c[p21], &c[p22], next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    sub<<<grid, block, 0, s0>>>(&a[p12], &a[p22], &c[p12]);
    add<<<grid, block, 0, s1>>>(&b[p21], &b[p22], &c[p21]);
    cudaThreadSynchronize();
    strassen(&c[p12], &c[p21], &c[p11], next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    add<<<grid, block, 0, s0>>>(&a[p11], &a[p22], &c[p12]);
    add<<<grid, block, 0, s1>>>(&b[p11], &b[p22], &c[p21]);
    cudaThreadSynchronize();
    strassen(&c[p12], &c[p21], t1, next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    wadd<<<grid, block, 0, s0>>>(t1, &c[p11], &c[p22]);
    add<<<grid, block, 0, s1>>>(&a[p21], &a[p22], t2);
    cudaThreadSynchronize();
    strassen(t2, &b[p11], &c[p21], next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    msub<<<grid, block, 0, s0>>>(&c[p21], &c[p22]);
    sub<<<grid, block, 0, s1>>>(&b[p21], &b[p11], t1);
    cudaThreadSynchronize();
    strassen(&a[p22], t1, t2, next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    wadd<<<grid, block, 0, s0>>>(t2, &c[p11], &c[p21]);
    sub<<<grid, block, 0, s1>>>(&b[p12], &b[p22], t1);
    cudaThreadSynchronize();
    strassen(&a[p11], t1, &c[p12], next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    madd<<<grid, block, 0, s0>>>(&c[p12], &c[p22]);
    add<<<grid, block, 0, s1>>>(&a[p11], &a[p12], t2);
    cudaThreadSynchronize();
    strassen(t2, &b[p22], t1, next, halfn, handle, alpha, beta, s0, s1);
    cudaThreadSynchronize();

    wsub_add<<<grid, block, 0, s0>>>(t1, &c[p11], &c[p12]);
  }
}

void call_strassen(float *a, float *b, float *c, float *t, int n){
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0, beta = 0.0;
  strassen(a, b, c, t, n, handle, &alpha, &beta, stream0, stream1);

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  cublasDestroy(handle);
}

void printm(float *a){
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < 4; j++){
      printf("%f ", a[i*N+j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(void){
  float *a, *b, *c1, *c2, *c3;
  float *dev_a, *dev_b, *dev_c, *dev_c2,*t;
  struct timeval t0, t1;

  cudaFuncSetCacheConfig(mmul, cudaFuncCachePreferShared);

  a = (float *)malloc(N*N*sizeof(float));
  b = (float *)malloc(N*N*sizeof(float));
  c1 = (float *)malloc(N*N*sizeof(float));
  c2 = (float *)malloc(N*N*sizeof(float));
  c3 = (float *)malloc(N*N*sizeof(float));

  HANDLE_ERROR(cudaMalloc((void**)&dev_a, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c2, tu1*tu1*sizeof(float)));
  HANDLE_ERROR(cudaMalloc((void**)&t, N*N*sizeof(float)));
  HANDLE_ERROR(cudaMemset(dev_c, 0, N*N*sizeof(float)));

  GPU_fill_rand(dev_a, N, N);
  GPU_fill_rand(dev_b, N, N);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(N/BLOCK_SIZE, N/BLOCK_SIZE);

  gettimeofday(&t0, NULL);
  mmul<<<grid, block>>>(dev_a, dev_b, dev_c, N);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(mmul)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));
  HANDLE_ERROR(cudaMemcpy(c1, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  gettimeofday(&t0, NULL);
  call_strassen(dev_a, dev_b, dev_c, t, N);
  cudaThreadSynchronize();
  gettimeofday(&t1, NULL);
  printf("Elapsed time(strassen)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));
  HANDLE_ERROR(cudaMemcpy(c2, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  gettimeofday(&t0, NULL);
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0, beta = 0.0;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_b, N, dev_a, N, &beta, dev_c, N);
  cudaThreadSynchronize();
  cublasDestroy(handle);
  gettimeofday(&t1, NULL);
  printf("Elapsed time(cublas)= %.10lf\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_usec - t0.tv_usec)/(1000.0*1000.0));

  HANDLE_ERROR(cudaMemcpy(c3, dev_c, N*N*sizeof(float), cudaMemcpyDeviceToHost));

  free(a);
  free(b);
  free(c1);
  free(c2);
  free(c3);

  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
  HANDLE_ERROR(cudaFree(dev_c2));
  HANDLE_ERROR(cudaFree(t));

  return 0;
}
