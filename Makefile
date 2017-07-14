strassen:
	nvcc -lcublas -lcurand -O3 -o strassen strassen.cu

clean:
	rm -f strassen
