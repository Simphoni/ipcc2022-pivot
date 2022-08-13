## Special case solution for k=2
时间：0726 1100
文件: opt2d.h opt2d.c  
编译: 
```makefile
CC = g++
CFLAGS=-lm -Ofast -fopenmp -O3

all: pivot

pivot: pivot.c opt2d.c opt2d.h
	$(CC) $^ $(CFLAGS) -o $@
clean:
	rm pivot
perf:
	srun -p IPCC ./pivot
```
调用方式: 见 pivot-xjz/pivot.c line 218
```cpp
solve2d(n, dim, coord, maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);
```
时间: n=500,k=2: 185 ms  
其他参数：使用64线程

时间：0807 2109
文件：opt2d.c solver.h commun.cpp
运行：n=500,k=2：86 ms
使用双机，各自启动42线程

时间：0812 0031
文件：opt5d.c
运行：n=100,k=5: 3080 ms
