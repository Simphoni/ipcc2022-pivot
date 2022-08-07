CC = mpicc
CFLAGS = -Ofast -O3 -fopenmp -mavx2 -mavx -march=native -lm -lstdc++
SRC := main.cpp opt4d.cpp opt2d.cpp commun.cpp

all: main

main: $(SRC)
	$(CC) $^ $(CFLAGS) -o $@
clean:
	rm main
perf2:
	srun -pIPCC -t 1 -N 2 -n 2 numactl -N 0,1 --membind=0,1 ./main
	diff -sb result.txt refer-2dim-5h.txt
perf4:
	srun -pIPCC -t 1 -N 2 -n 2 numactl -N 0,1 --membind=0,1 ./main uniformvector-4dim-1h.txt
	diff -sb result.txt refer-4dim-1h.txt
