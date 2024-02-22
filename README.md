# Fault-simulation

## Build adn run

```
~$ mkdir build
~$ cd build
~$ cmake ../
~$ make
~$ ./main/simple ${benchmark.ckt} ${benchmark.flst} ${benchmark.ptn} # for the CPU version
~$ ./main/simple_cuda ${benchmark.ckt} ${benchmark.flst} ${benchmark.ptn} # for the GPU version

```

* Files
	- .ckt: circuit file
	- .flst: fault list file
	- .ptn: pattern list file


## Records
* CudaManaged memory version - commit: ADD large circuits input - random2 (sz: b18), random3 (sz: b19)
	* Pros: 
		* This version stored all of the data on the unified memory. Therefore, we don't need to worry the limitation memory of the GPU that cannot store all of the data on the GPU memory.
	* Cons: 
		* I/O bottleneck, since all thread need to go back to CPU to get their data all the time $\to$ also extremely time comsuing. 

