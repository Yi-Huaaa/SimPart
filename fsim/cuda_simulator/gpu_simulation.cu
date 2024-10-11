#include "gpu_simulation.cuh"
// #define TI template<typename IndexType> 



__global__ void BIBI_TRYTRY(const int a) {
  printf("BIBI_TRYTRY, a = %d\n", a);
  bibi_trytry(a+10);
  bibi_trytry(a+20);
  bibi_trytry(a+30);
  bibi_trytry(a+40);
}

__device__ void bibi_trytry(const int a) {
  printf("bibi_trytry, a = %d\n", a);

}



// // ------------------ Testing functions above  ------------------ 



// void GALPS_GPUSimulator::_gpu_ask_copy_mem() {
//   // needed data 
//   cudaMalloc((void**)&_pi_gate_po_gate_type_gpu, (_sum_pi_gates_pos)*sizeof(size_t));
//   cudaMalloc((void**)&_pi_gate_po_output_res_gpu, (_sum_pi_gates_pos)*sizeof(size_t));
//   cudaMalloc((void**)&_pi_gate_po_num_inputs_gpu, (_sum_pi_gates_pos)*sizeof(size_t));
//   cudaMalloc((void**)&_patterns_gpu, (_num_rounds*_num_PIs)*sizeof(size_t));
//   cudaMalloc((void**)&_fault_gate_idx_gpu, _num_faults*sizeof(size_t));
//   cudaMalloc((void**)&_fault_SA_fault_val_gpu, _num_faults*sizeof(size_t));
  
//   cudaMalloc((void**)&_g_pi_results_gpu, _num_PIs*sizeof(size_t));
//   cudaMalloc((void**)&_g_gate_results_gpu, _num_inner_gates*sizeof(size_t));
//   cudaMalloc((void**)&_g_po_results_gpu, _num_POs*sizeof(size_t));
//   cudaMalloc((void**)&_b_pi_results_gpu, _num_PIs*sizeof(size_t));
//   cudaMalloc((void**)&_b_gate_results_gpu, _num_inner_gates*sizeof(size_t));
//   cudaMalloc((void**)&_b_po_results_gpu, _num_POs*sizeof(size_t));
//   cudaMalloc((void**)&_found_fault_to_pattern_gpu, 2*_num_faults*sizeof(size_t));
  
//   cudaMalloc((void**)&_gate_inputs_list_gpu, _sum_inputs_gates*sizeof(size_t));
//   cudaMalloc((void**)&_gate_inputs_list_start_gpu, _sum_pi_gates_pos*sizeof(size_t));

//   cudaMalloc((void**)&_num_gates_per_level_gpu, _total_num_levels*sizeof(size_t));
//   size_t orders_size = (_pull_up_head_frt_optimization) ? 
//             (_num_total_gates_after_head_end_opt) : (_sum_pi_gates_pos);
//   cudaMalloc((void**)&_order_gpu, orders_size*sizeof(size_t));


//   // copy memory 
//   cudaMemcpy(_gate_inputs_list_gpu, _gate_inputs_list, _sum_inputs_gates*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaCheckErrors("_gate_inputs_list_gpu failure");
//   cudaMemcpy(_gate_inputs_list_start_gpu, _gate_inputs_list_start, _sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaCheckErrors("_gate_inputs_list_start_gpu failure");

//   // prepare data for CSR format: 
//   // pi_gate_po_gate_type_gpu
//   size_t *pi_gate_po_gate_type = (size_t *)malloc(_sum_pi_gates_pos*sizeof(size_t));
//   size_t *pi_gate_po_num_inputs = (size_t *)malloc(_sum_pi_gates_pos*sizeof(size_t));
//   memset(pi_gate_po_gate_type, 0, (_sum_pi_gates_pos)*sizeof(size_t));
//   memset(pi_gate_po_num_inputs, 0, (_sum_pi_gates_pos)*sizeof(size_t));
//   for (size_t i = 0; i < _num_PIs; i++) {
//     pi_gate_po_gate_type[i] = static_cast<size_t>(_PIs[i]._type);
//     pi_gate_po_num_inputs[i] = _PIs[i]._num_inputs;
//   }
//   for (size_t i = 0; i < _num_inner_gates; i++) {
//     pi_gate_po_gate_type[i+_num_PIs] = static_cast<size_t>(_gates[i]._type);
//     pi_gate_po_num_inputs[i+_num_PIs] = _gates[i]._num_inputs;
//   }
//   for (size_t i = 0; i < _num_POs; i++) {
//     pi_gate_po_gate_type[i+_num_PIs+_num_inner_gates] = static_cast<size_t>(_POs[i]._type);
//     pi_gate_po_num_inputs[i+_num_PIs+_num_inner_gates] = _POs[i]._num_inputs;
//   }

//   // _patterns_gpu
//   size_t *_patterns_cpu = (size_t *)malloc((_num_rounds*_num_PIs)*sizeof(size_t));
//   for (size_t i = 0; i < _num_rounds; i++) {
//     for (size_t pi = 0; pi < _num_PIs; pi++) {
//       size_t idx = _num_PIs*i + pi;
//       _patterns_cpu[idx] = _patterns[i]._value[pi];
//     }
//   }
  
//   // _fault_gate_idx_gpu; _fault_SA_fault_val_gpu;
//   size_t *fault_gate_idx_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
//   size_t *fault_SA_fault_val_cpu = (size_t *)malloc((_num_faults)*sizeof(size_t));
//   for (size_t i = 0; i < _num_faults; i++) {
//     fault_gate_idx_cpu[i]     = _faults[i]._gate_with_fault;
//     fault_SA_fault_val_cpu[i] = _faults[i]._gate_SA_fault_val;
//   }
//   cudaCheckErrors("fault_gate_idx cpy failure");


//   cudaMemcpy(_pi_gate_po_gate_type_gpu, pi_gate_po_gate_type, _sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(_pi_gate_po_num_inputs_gpu, pi_gate_po_num_inputs, _sum_pi_gates_pos*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(_patterns_gpu, _patterns_cpu, (_num_rounds*_num_PIs)*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(_fault_gate_idx_gpu, fault_gate_idx_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(_fault_SA_fault_val_gpu, fault_SA_fault_val_cpu, _num_faults*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaMemcpy(_num_gates_per_level_gpu, _num_gates_per_level, _total_num_levels*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaCheckErrors("cudaMemcpy 5 failure");

//   cudaMemcpy(_order_gpu, _order_cpu, orders_size*sizeof(size_t), cudaMemcpyHostToDevice);
//   cudaCheckErrors("cudaMemcpy 6 failure");
  
  
//   free(pi_gate_po_gate_type);
//   free(pi_gate_po_num_inputs);  
//   free(_patterns_cpu);
//   free(fault_gate_idx_cpu);
//   free(fault_SA_fault_val_cpu);
//   free(_order_cpu);
  
// #ifdef FSIM_DEBUG_GPU_INPUTS
//   // check 
//   // _pi_gate_po_gate_type_gpu
//   print_pi_gate_po_gate_type_gpu <<< 1, 1 >>> (_pi_gate_po_gate_type_gpu, _pi_gate_po_num_inputs_gpu, _sum_pi_gates_pos);

//   // _patterns_gpu
//   print_patterns_gpu <<< 1, 1 >>> (_patterns_gpu, _num_rounds, _num_PIs);

//   // _fault_gate_idx_gpu && _FFR_levelization_gpu
//   print_fault_gate_idx_gpu <<< 1, 1 >>> (_fault_gate_idx_gpu, _fault_SA_fault_val_gpu, _num_faults, 
//                                         _FFR_levelization_gpu, 0, _total_num_levels-_start_tail_end_level);

// #endif
// }




// void GALPS_GPUSimulator::_run_levelized_gpu() {
// #ifdef GALPS_DEBUG_PRINT_SIMULATION
//   std::cout << "execute simulation._run_parallel cuda();" << '\n';
// #endif

//   // ask memory 
//   _gpu_ask_copy_mem();

//   for (size_t rd = 0; rd < _num_rounds; rd++) {
//     size_t num_testcases_this_round =
//         ((_num_pattern / (SIZE_T_BITS * (rd + 1))))
//             ? (SIZE_T_BITS)
//             : (_num_pattern % SIZE_T_BITS);

//     // good simulation
// #ifdef GALPS_DEBUG_PRINT_SIMULATION
//     std::cout << "CUDA: Run Good case" << std::endl;
// #endif

//     _run_good_case_gpu(rd, num_testcases_this_round);

//     // bad simulation (fault simulation)
//     for (size_t j = 0; j < _num_faults; j++) {
//       _run_bad_case_cuda(rd, j, num_testcases_this_round);
//       _compare_g_b_results <<< 1, 1 >>> (j, _num_POs, rd, _g_po_results_gpu, _b_po_results_gpu, _found_fault_to_pattern_gpu);
//     }
//   }
  
//   // TODO: free memory 
// }







// void GALPS_GPUSimulator::_run_good_case_gpu(const size_t rd, const size_t bits) {
// #ifdef GALPS_DEBUG_PRINT_SIMULATION
//   printf("insides _run_good_case_gpu\n");
// #endif  
  
//   size_t num_blocks, num_threads;
//   const size_t fault_num = 0; const size_t bad_case = 0;
//   size_t accum = 0;

//   // Simulation
//   num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
//   num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs); // _num_PIs;
//   // printf("num_blocks = %ld, num_threads = %ld\n", num_blocks, num_threads);
  

//   // since the level_0 belongs to PI, the level_MAX belongs to POs
//   accum += _num_PIs;
//   for (size_t i = 1; i < (_total_num_levels - 1); i++) {
    
//     num_blocks = (_num_gates_per_level[i] > 1024) ? (_num_gates_per_level[i] + 1024 - 1) >> 10 : (1);//1;
//     num_threads = (_num_gates_per_level[i] > 1024) ? (1024) : (_num_gates_per_level[i]);// _num_gates_per_level[i];
//     _run_gate <<< num_blocks, num_threads >>> (_gate_inputs_list_gpu, _gate_inputs_list_start_gpu, 
//                                               _pi_gate_po_gate_type_gpu, _pi_gate_po_output_res_gpu, 
//                                               _pi_gate_po_num_inputs_gpu, _patterns_gpu, 
//                                               _fault_gate_idx_gpu, _fault_SA_fault_val_gpu, 
//                                               rd, fault_num, bad_case, _num_gates_per_level[i], accum, _num_PIs);  
//     accum += _num_gates_per_level[i];
//   }


//   // shift and copy answer to the good results
//   if (bits < SIZE_T_BITS) {
//     num_blocks = (_num_PIs > 1024) ? ((_num_PIs + 1024 - 1) >> 10) : (1); // 1;  
//     num_threads = (_num_PIs > 1024) ? (1024) : (_num_PIs);     
//     _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_pi_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, 0, _num_PIs);
    
//     num_blocks = (_num_inner_gates > 1024) ? (_num_inner_gates + 1024 - 1) >> 10 : (1);//1;
//     num_threads = (_num_inner_gates > 1024) ? (1024) : (_num_inner_gates);    
//     _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_gate_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs, _num_inner_gates);

//     num_blocks = (_num_POs > 1024) ? ((_num_POs + 1024 - 1) >> 10) : (1); // 1;
//     num_threads = (_num_POs > 1024) ? (1024) : (_num_POs);    
//     _shift_to_correct_answer_gpu <<< num_blocks, num_threads >>> (_g_po_results_gpu, _pi_gate_po_output_res_gpu, bits, SIZE_T_BITS, _num_PIs+_num_inner_gates, _num_POs);
//   }

// #ifdef GALPS_DEBUG_PRINT_SIMULATION
//   cudaDeviceSynchronize();  
//   printf("---\n");
// #endif  
// }


// __global__ void _run_gate(size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu, 
//                           size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_output_res_gpu, 
//                           size_t *_pi_gate_po_num_inputs_gpu, size_t *_patterns_gpu, 
//                           size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, 
//                           const size_t rd, const size_t fault_num, const size_t bad_case, 
//                           const size_t max_gate_num_per_level, const size_t fin_num_gates, const size_t _num_PIs) {  
//   size_t t_idx = threadIdx.x;
//   size_t b_idx = blockIdx.x;
//   size_t real_g_idx = blockDim.x * b_idx + t_idx + fin_num_gates; // tmp
//   if (real_g_idx > (max_gate_num_per_level+fin_num_gates))
//     return;
  
//   size_t SA_fault = (((real_g_idx) == _fault_gate_idx_gpu[fault_num]) & bad_case);
  
//   if (SA_fault) {
//     _pi_gate_po_output_res_gpu[real_g_idx] = _fault_SA_fault_val_gpu[fault_num];
//     return;
//   }

//   size_t type = _pi_gate_po_gate_type_gpu[real_g_idx];
//   size_t num_inputs = _pi_gate_po_num_inputs_gpu[real_g_idx];
//   switch (type) { 
//     case 0:
//       _apply_INV(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 1:
//       _apply_AND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 2:
//       _apply_OR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 3:
//       _apply_XOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 4:
//       _apply_NAND(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 5:
//       _apply_NOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 6:
//       _apply_XNOR(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 7:
//       _apply_MUX(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 8:
//       _apply_CLKBUF(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 9:
//       _apply_PI(real_g_idx, _pi_gate_po_output_res_gpu, _patterns_gpu[_num_PIs*rd+real_g_idx]);
//     break;
//     case 10:
//       _apply_PO(real_g_idx, num_inputs, _pi_gate_po_output_res_gpu, _gate_inputs_list_gpu, _gate_inputs_list_start_gpu);
//     break;
//     case 11:
//     break;
//     }
// }


// // simulation for gates
// __device__ __forceinline__ void _apply_INV(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx]; // pass in real index
//   size_t pre_gate_idx = _gate_inputs_list_gpu[start_location];

//   _pi_gate_po_output_res_gpu[gate_idx] = ~(_pi_gate_po_output_res_gpu[pre_gate_idx]);
// }

// __device__ __forceinline__ void _apply_AND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret &= now_gate_val;     
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ret;
// }

// __device__ __forceinline__ void _apply_OR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret |= now_gate_val; 
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ret;
// }

// __device__ __forceinline__ void _apply_XOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret ^= now_gate_val; 
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ret;
// }

// __device__ __forceinline__ void _apply_NAND(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret &= now_gate_val; 
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);
// }

// __device__ __forceinline__ void _apply_NOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret |= now_gate_val; 
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);   
// }

// __device__ __forceinline__ void _apply_XNOR(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t ret = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];

//   for (size_t i = 1; i < num_inputs; i++) {
//     size_t now_gate_val =_pi_gate_po_output_res_gpu[ _gate_inputs_list_gpu[start_location+i]];
//     ret ^= now_gate_val; 
//   }
//   _pi_gate_po_output_res_gpu[gate_idx] = ~(ret);
// }

// __device__ __forceinline__ void _apply_MUX(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   size_t a = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+0]];
//   size_t b = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+1]];
//   size_t s = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location+2]];

//   size_t ret = ((s & b) | ( a & (!s)));

//   _pi_gate_po_output_res_gpu[gate_idx] = (ret);
// }

// __device__ __forceinline__ void _apply_CLKBUF(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];
// }

// __device__ __forceinline__ void _apply_PI(size_t gate_idx, size_t *_pi_gate_po_output_res_gpu, size_t pattern_val) {
//   _pi_gate_po_output_res_gpu[gate_idx] = pattern_val; 
// }

// __device__ __forceinline__ void _apply_PO(size_t gate_idx, size_t num_inputs, size_t *_pi_gate_po_output_res_gpu, size_t *_gate_inputs_list_gpu, size_t *_gate_inputs_list_start_gpu) {
//   size_t start_location = _gate_inputs_list_start_gpu[gate_idx];
//   _pi_gate_po_output_res_gpu[gate_idx] = _pi_gate_po_output_res_gpu[_gate_inputs_list_gpu[start_location]];
// }

// __global__ void _show_computational_results (size_t *a, size_t start, size_t end) {
//   printf("_show_computational_results\n");

//   for (size_t i = start; i < end; i++) {
//     printf("arr[%lu] = %lu\n", i, a[i]);
//   }
//   printf("\n");
// }


// __global__ void _compare_g_b_results(const size_t j, const size_t _num_POs, const size_t rd, 
//   size_t *_g_po_results_gpu, size_t *_b_po_results_gpu, size_t *_found_fault_to_pattern_gpu) {

//   size_t found_fault = 0;
//   for (size_t i = 0; i < _num_POs; i++) {
//     if (_g_po_results_gpu[i] != _b_po_results_gpu[i]) {
//       found_fault = 1;
//     break;
//     }
//   }
//   // Record whether fault can be found
//   _found_fault_to_pattern_gpu[2 * j] = found_fault;
//   // Record which pattern found the fault
//   _found_fault_to_pattern_gpu[2 * j + 1] = rd;
// }

// __global__ void print_pi_gate_po_gate_type_gpu(size_t *_pi_gate_po_gate_type_gpu, size_t *_pi_gate_po_num_inputs_gpu, size_t num) {
//   printf("print_pi_gate_po_gate_type_gpu (num = %ld): [\n", num);
//   for (size_t i = 0; i < num; i++) {
//     size_t type = _pi_gate_po_gate_type_gpu[i];
//     switch (type) {
//       case 0:
//         printf("i = %ld, type_INV\n", i);
//         break;
//       case 1:
//         printf("i = %ld, type_AND\n", i);
//         break;
//       case 2:
//         printf("i = %ld, type_OR\n", i);
//         break;
//       case 3:
//         printf("i = %ld, type_XOR\n", i);
//         break;
//       case 4:
//         printf("i = %ld, type_NAND\n", i);
//         break;
//       case 5:
//         printf("i = %ld, type_NOR\n", i);
//         break;
//       case 6:
//         printf("i = %ld, type_XNOR\n", i);
//         break;
//       case 7:
//         printf("i = %ld, type_MUX\n", i);
//         break;
//       case 8:
//         printf("i = %ld, type_CLKBUF\n", i);
//         break;
//       case 9:
//         printf("i = %ld, type_PI\n", i);
//         break;
//       case 10:
//         printf("i = %ld, type_PO\n", i);
//         break;
//       case 11:
//         printf("i = %ld, type_UNKNOWN\n", i);
//         break;
//     }
//   }
//   printf("]\n");

//   printf("_pi_gate_po_num_inputs_gpu =[\n");  
//   for (size_t i = 0; i < num; i++) {
//     printf("%ld, ", _pi_gate_po_num_inputs_gpu[i]);
//   } printf("]\n");
// }
// __global__ void print_patterns_gpu(size_t *_patterns_gpu, size_t _num_rounds, size_t _num_PIs) {
//   printf("print_patterns_gpu (_num_rounds = %ld, _num_PIs = %ld): [\n", _num_rounds, _num_PIs);
//   for (size_t i = 0; i < _num_rounds; i++) {
//     printf("RD_%ld: [", i);
//     for (size_t pi = 0; pi < _num_PIs; pi++) {
//       size_t idx = _num_PIs*i + pi;
//       printf("%ld, ", _patterns_gpu[idx]);
//     } printf("]\n");
//   } printf("]\n");
// }
// __global__ void print_fault_gate_idx_gpu(size_t *_fault_gate_idx_gpu, size_t *_fault_SA_fault_val_gpu, size_t _num_faults, 
//                                           size_t * _FFR_levelization_gpu, size_t s, size_t n) {
//   printf("print_fault_gate_idx_gpu (_num_faults = %ld): [\n", _num_faults);
//   for (size_t i = 0; i < _num_faults; i++) {
//     printf("gate_w_f = %ld, SA_val = %ld\n", _fault_gate_idx_gpu[i], _fault_SA_fault_val_gpu[i]);
//   } printf("]\n");

//   printf("_FFR_levelization_gpu =\n");
//   for (size_t i = s; i < n; i++) {
//     printf("!_FFR_levelization_gpu[%ld] = %ld\n", i, _FFR_levelization_gpu[i]);
//   } printf("]\n");
// }


// void GALPS_GPUSimulator::_free_all() {  
//   cudaFree(_pi_gate_po_gate_type_gpu);
//   cudaFree(_pi_gate_po_output_res_gpu);

//   cudaFree(_patterns_gpu);
//   cudaFree(_patterns_gpu);

//   cudaFree(_fault_gate_idx_gpu);
//   cudaFree(_fault_SA_fault_val_gpu);

//   cudaFree(_g_pi_results_gpu);
//   cudaFree(_g_gate_results_gpu);
//   cudaFree(_g_po_results_gpu);
//   cudaFree(_b_pi_results_gpu);
//   cudaFree(_b_gate_results_gpu);
//   cudaFree(_b_po_results_gpu);
//   cudaFree(_found_fault_to_pattern_gpu);
  
//   cudaFree(_gate_inputs_list_gpu);
//   cudaFree(_gate_inputs_list_start_gpu); 

//   cudaFree(_params_gpu);
// }

