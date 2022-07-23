CC = g++
CFLAGS=-lm -Ofast -fopenmp -O3

all: pivot

pivot: pivot.c opt2d.c opt2d.h
	$(CC) $^ $(CFLAGS) -o $@
clean:
	rm pivot
perf:
	srun -p IPCC numactl --interleave=all ./pivot
