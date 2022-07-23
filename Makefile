CC = gcc
CFLAGS=-lm -Ofast -fopenmp

all: pivot

pivot: pivot.c opt2d.c opt2d.h
	$(CC) $^ $(CFLAGS) -o $@
clean:
	rm pivot
