# Matrix Multiplicaion of Strassen algorithm on GPU
### What is it?
strassen.cu is cuda program calculating matrix multiplication and compare the time of 3 algorithm.
+ mmul uses naive algorithm using shared memory to calculate faster.
+ strassen uses strassen algorithm.

Reference:  https://www.cise.ufl.edu/~sahni/papers/strassen.pdf.

When N = 16384 matrix, the strassen algorithm could calculate faster than cublasSgemm.

### Run this program
$ make

$ ./strassen

### The result
GPU : NVIDIA Quadro M4000

|      |mmul | Strassen | cublasSgemm |
|:-----|:-----:|:--------:|------------:|
| N=8192| 2.858144(sec) | 0.559933(sec) | 0.488848(sec) |
| N=16384 | 22.80154(sec) | 3.364058(sec) | 3.88669(sec) |

Note: An include file "book.h" is a sample program in https://developer.nvidia.com/cuda-example. So if you want to compile this program, you havet to download this program and compile together.
