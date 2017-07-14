export CUDA_PROFILE=1
nvcc -lcublas -lcurand -O3 -o strassen strassen.cu

export CUDA_PROFILE_CONFIG=prof.conf
export CUDA_PROFILE_LOG=a.log
./strassen

export CUDA_PROFILE=0
