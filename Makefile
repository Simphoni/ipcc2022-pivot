CFLAGS=-lm -Ofast

all: pivot

pivot: pivot.c
	gcc pivot.c $(CFLAGS) -o pivot
