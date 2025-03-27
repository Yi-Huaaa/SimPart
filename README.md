# Fault-simulation

## Build and run

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

## Nsight profile (NCU)
```
sudo /usr/local/cuda-12.3/bin/ncu --set full -o temp.ncu-rep --force-overwrite --kill on --target-processes application-only --replay-mode kernel --kernel-name-base function --kernel-name _run_gate_graph_partition --launch-skip 0 --launch-count 2 --warp-sampling-interval auto --warp-sampling-max-passes 1 --warp-sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --check-exit-code yes --import-source yes --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \./main/simple_cuda "../benchmark/large_circuits/mgc_edit_dist_iccad_dfs.ckt" "../benchmark/large_circuits/mgc_edit_dist_iccad.flst" "../benchmark/large_circuits/mgc_edit_dist_iccad.ptn" "visualization_graph.txt"

```

### 20241024 update command
```
sudo /usr/local/cuda-12.3/bin/ncu --set full -o 1024.ncu-rep --force-overwrite --kill on --target-processes application-only --replay-mode kernel --kernel-name-base function --kernel-name _run_gate_part --launch-skip 0 --launch-count 2 --warp-sampling-interval auto --warp-sampling-max-passes 1 --warp-sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --check-exit-code yes --import-source yes --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \./main/simple_cuda_partition "../benchmark/large_circuits/mgc_edit_dist_iccad_bfs.ckt" "../benchmark/large_circuits/mgc_edit_dist_iccad.flst" "../benchmark/large_circuits/mgc_edit_dist_iccad.ptn" "visualization_graph.txt"
```

## 20250321 update command
```
sudo /usr/local/cuda-12.3/bin/ncu --set full -o 0321.ncu-rep --force-overwrite --kill on --target-processes application-only --replay-mode kernel --kernel-name-base function --kernel-name _run_gate_MA --launch-skip 0 --launch-count 20 --warp-sampling-interval auto --warp-sampling-max-passes 1 --warp-sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --check-exit-code yes --import-source yes --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats \./main/MA_cuda_partition "../benchmark/large_circuits/mgc_edit_dist_iccad_bfs.ckt" "../benchmark/large_circuits/mgc_edit_dist_iccad.flst" "../benchmark/large_circuits/mgc_edit_dist_iccad.ptn" "../benchmark/pre_construct_hypergraph/ok_inputs/mgc_edit_dist_iccad_output.txt" "visualization_graph.txt" 1
```




## Records
* CudaManaged memory version - commit: ADD large circuits input - random2 (sz: b18), random3 (sz: b19)
	* Pros: 
		* This version stored all of the data on the unified memory. Therefore, we don't need to worry the limitation memory of the GPU that cannot store all of the data on the GPU memory.
	* Cons: 
		* I/O bottleneck, since all thread need to go back to CPU to get their data all the time $\to$ also extremely time comsuing. 

