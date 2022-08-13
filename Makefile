CC = mpicxx
CFLAGS = -Ofast -O3 -fopenmp -mavx2 -mavx -march=native -lm -lstdc++
SRC := main.cpp opt5d.cpp opt2d.cpp commun.cpp

all: main

main: $(SRC)
	$(CC) $^ $(CFLAGS) -o $@
clean:
	rm main
perf2:
	srun -pIPCC -t 1 -N 2 -n 2 numactl -N 0,1 --membind=0,1 ./main
perf4:
	env OMP_PROC_BIND=close env OMP_PLACES=cores srun -pIPCC -t 1 -N 2 -n 2 ./main uniformvector-4dim-1h.txt
check2:
	diff -sb result.txt refer-2dim-5h.txt
check4:
	diff -sb result.txt refer-4dim-1h.txt
