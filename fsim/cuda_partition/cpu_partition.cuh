#ifndef CPU_PARTITION_H
#define CPU_PARTITION_H

#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include <list>
#include <cublas_v2.h>
#include <set>

// CPU
#include <fsim/fsim.hpp>
#include <libkahypar.h> // kahypar single thread
#include <libmtkahypar.h> // kahypar multi-threads

#include "cuda_partition.cuh"


#define ELEMENT_INDEX_TYPE size_t
#define ELEMENT_LEVEL_TYPE size_t
#define FAULT_INDEX_TYPE size_t



// Forward declaration
class CUDASimulator;
class GALPS_SIZET_GPUSimulator;

class CPUPartitioner {
  
  // Declare friend class 
  friend class CUDASimulator;
  friend class GALPS_SIZET_GPUSimulator;

public:
  enum class Mode { SEQUENTIAL = 0, 
                    PARALLEL_OMP = 1, 
                    REPCUT_PARALLEL_OMP = 2, 
                    MT_REPCUT_PARALLEL_OMP = 3,
                  };
  enum class SIMU_Mode {CPU = 0, 
                        GPU = 1,
                        };


  // accessor
  void read(Mode mode, const std::string &ckt_path, 
            const std::string &flst_path, 
            const std::string &ptn_path,
            const std::string &hyg_path,
            const size_t num_threads,
            const int hypergraph_construct_mode);
  void read(Mode mode, std::istream &ckt, std::istream &flst, std::istream &ptn, std::istream &hyg, 
            const size_t num_threads,
            const int hypergraph_construct_mode);
  void read_pre_construct_HG(Mode mode, const std::string &ckt_path, 
                              const std::string &flst_path, 
                              const std::string &ptn_path,
                              const std::string &hyg_path,
                              const size_t num_threads);
  void read_pre_construct_HG(Mode mode, std::istream &ckt, std::istream &flst, std::istream &ptn, std::istream &hyg, 
                              const size_t num_threads);
  void run(SIMU_Mode simu_mode, Mode mode, const size_t num_threads, const size_t NUM_SIMULATION_RDS);
  
  void visualization_graph(const std::string &outputFile);
  void KaHyParExample();
  void mt_KaHyParExample();

  void prepare_cpu_simulation(const int hypergraph_construct_mode, Mode mode);
  void prepare_gpu_simulation(Mode mode, const int hypergraph_construct_mode);

private:
  /* Private members */
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t _sum_pi_gates_pos;
  std::vector<ElementBase<ELEMENT_INDEX_TYPE, ELEMENT_LEVEL_TYPE>> _Gates;
  std::vector<std::vector<size_t>> _adj;
  std::vector<std::vector<size_t>> _invAdj;
  std::vector<GateType> _gate_type;

  bool _preconstruct_hypergraph_print_flag = false;
  std::vector<size_t> _vertex_to_partition;


  // Fault and Pattern 
  size_t _num_fault; // for read file
  std::vector<Fault<FAULT_INDEX_TYPE>> _faults;

  size_t _num_pattern; // total number of patterns that need to be tested
  size_t _num_rounds;  // ceiling(_num_pattern/SIZE_T_BITS)
  std::vector<Pattern> _patterns;

  // For levelization and Find cones
  std::vector<size_t> _sources; 
  std::vector<size_t> _sinks; 
  size_t _max_level;
  size_t _total_num_levels;
  std::vector<size_t> _level_of_gates;
  std::vector<std::vector<size_t>> _gateIdx_in_each_level;
  std::vector<std::vector<std::vector<size_t>>> _cones;

  // Simulation
  std::vector<size_t> _g_pi_results;
  std::vector<size_t> _g_gate_results;
  std::vector<size_t> _g_po_results;
  std::vector<size_t> _b_pi_results;
  std::vector<size_t> _b_gate_results;
  std::vector<size_t> _b_po_results;
  std::vector<size_t> _found_fault_to_pattern;

  // For merge
  // const size_t _merge_threshold = 11; // adjustable  
  // const size_t _merge_bound = 2;


  // For repCut
  // TODO: lots of things have no need to be private members -> move back to local 
  bool _can_run_kahypar = true;
  std::vector<std::vector<size_t>> _clusters; 
  std::vector<size_t> _to_hypergraph_vertex_idx; // cluster index to (1) vectex index and (2) hyperedge index 
  std::vector<std::vector<size_t>> _gate_to_cones; 
  std::vector<size_t> _gate_to_cluster; 
  std::vector<size_t> _is_sink_clusters_idx; 
  std::vector<size_t> _non_sink_clusters_idx;
  std::vector<bool> _is_sink_clusters; // bool is sink cluster or now
  std::vector<size_t> _vertexIdx_to_sinkGateIdx; 
  std::vector<size_t> _sinkGateIdx_to_vertexIdx; 
  std::vector<std::vector<std::set<size_t>>> _RepCut_output_cones_set;
  std::vector<std::vector<std::vector<size_t>>> _RepCut_output_cones;

  // For KaHyPar
  std::vector<size_t> _vectices_weight;
  std::vector<size_t> _hyperedges_weight;
  std::vector<size_t> _hyperedges_eind;
  std::vector<size_t> _hyperedges_eptr;
  const double _imbalance = 0.015;
  const kahypar_partition_id_t _k = 16; // number of partitions 
  const size_t _k_sizet = 16; 
  size_t _KaHyPar_num_vertices = 0; // number of sinks
  size_t _KaHyPar_num_hyperedges = 0;

  // declare as member since mt-kahypar needs it
  std::vector<std::set<size_t>> _hyperedge_include_which_vertices;
  std::vector<std::set<size_t>> _vertex_include_which_hyperedges;


  // GPU simulation
  size_t *_pi_gate_po_gate_type_gpu;
  size_t *_patterns_gpu;
  size_t *_fault_gate_idx_gpu;
  size_t *_fault_SA_fault_val_gpu;
  size_t *_pi_gate_po_output_res_gpu;
  size_t *_g_pi_results_gpu;
  size_t *_g_gate_results_gpu;
  size_t *_g_po_results_gpu;
  size_t *_b_pi_results_gpu;
  size_t *_b_gate_results_gpu;
  size_t *_b_po_results_gpu;
  size_t *_found_fault_to_pattern_gpu;
  // TODO prepare
  std::vector<size_t> _numGates_per_level_gpu_of_groups; // partitioned cones CPU
  size_t *_numGates_per_level_gpu_of_groups_gpu;
  size_t *_cones_partitioned_gpu; // partitioned results 
  size_t *_invAdj_gpu; 
  size_t *_invAdj_index_table_gpu;
  size_t *_per_level_of_group_start_accum_gpu;




  /* Functions */
  // Construct
  void _read_graph(std::istream &ckt);
  void _levelization();
  

  void _partition();
  void _show_cones_size();
  bool _gateHasCone(std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    const size_t gateIdx);
  bool _gateHasPostGates(const size_t gateIdx, const size_t startLevel);         
  void _getConeIdices(const size_t conesSz, const size_t gateIdx, const size_t startLevel, 
                      bool newCone, 
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones);
  void _updateConIdxTable(bool newCone, const size_t gateIdx, 
                          std::vector<size_t> &coneIdxToSinkGateIdx);
  void _update_cones(bool newCone, const size_t gateIdx, 
                    const size_t startLevel, 
                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    size_t &coneWidth);
  void _find_all_cones(const size_t startLevel, 
                          std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                          std::vector<size_t> &coneIdxToSinkGateIdx,
                          int &stopLevel);
void _construct_duplication_table(std::vector<size_t> &coneIdxToSinkGateIdx,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable,
                      const size_t startLevel, const int stopLevel, 
                      const size_t numLevels);

  // RepCut
  void _get_hypergraph(const int mode, std::istream &hyg,
                      std::chrono::duration<double> &duration_load_data);
    void _load_constructed_hypergraph(std::istream &hyg);
  void _RepCut(std::chrono::duration<double> &duration_0);
  void _RepCut_MT(std::chrono::duration<double> &duration_0, 
                  const size_t num_threads);
  void _find_clusters();
  void _construct_hyperGraph();
  void _construct_hyperedge_vertices();
  void _compute_vertex_weight(std::vector<double> &tmp_vectices_weight,
        std::vector<std::set<size_t>> &vertex_include_which_hyperedges);
  void _multiple_100(std::vector<double> &tmp_vectices_weight);        
  void _run_KaHyPar();
  void _run_mtKaHyPar(const size_t num_threads);
  void _get_KHP_partitioning_res(const int hypergraph_construct_mode);

  void _construct_partitioned_groups();
    void _construct_groups();
    void _get_sinks_group(std::vector<std::vector<size_t>> &sinks_groups);
    void _construct_cones(std::vector<size_t> &sinks,
                          std::vector<std::set<size_t>> &_cones_set_g,
                          std::vector<bool> &visited_cpu);

  // Simulator
  void _run_cpu_simulator(Mode mode, const size_t num_threads);
  void _run_gpu_simulator(Mode mode, const size_t NUM_SIMULATION_RDS);

  // GPU Simulation
  void _ask_gpu_memory_simu_1(Mode mode);
  void _ask_gpu_memory_simu_2();
  void _move_GateType_h2d();
  void _move_patterns_h2d();
  void _move_faults_h2d();


  // CPU Simulation
  // FSIM
  void _construct_graph();
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);
  void _ask_simulation_memory() {
    _g_pi_results.resize(_num_PIs);
    _g_gate_results.resize(_num_inner_gates);
    _g_po_results.resize(_num_POs);
    _b_pi_results.resize(_num_PIs);
    _b_gate_results.resize(_num_inner_gates);
    _b_po_results.resize(_num_POs);
    _found_fault_to_pattern.resize(2 * _faults.size());    
  }
  void _write_to_array(std::vector<size_t> &pi_output, 
                      std::vector<size_t> &gate_output,
                      std::vector<size_t> &po_output,
                      const size_t bits);

  // Simulation - sequentail and parallel (without partitioning to cones)
  void _run_fsim(const size_t num_threads);
  void _run_good_case(const Pattern pattern, const size_t bits, const size_t num_threads);
  void _run_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, 
                    const Pattern pattern, 
                    const size_t bits, const size_t num_threads);
  
  // Simulation - sequentail and parallel (with partitioning to cones)
  void _run_cones_gates(const size_t num_threads);
  void _run_cones_good_case(const Pattern pattern, const size_t bits, const size_t num_threads);
  void _run_cones_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, 
                          const Pattern pattern, 
                          const size_t bits, const size_t num_threads);
  
  // Simulation - sequentail and parallel (with partitioning to RepCut's output cones)
  void _run_RepCut_cones_gates(const size_t num_threads);
  void _run_RepCut_cones_good_case(const Pattern pattern, 
                                  const size_t bits, 
                                  const size_t num_threads);
  void _run_RepCut_cones_bad_case(const Fault <FAULT_INDEX_TYPE> &fault, 
                                  const Pattern pattern, 
                                  const size_t bits, const size_t num_threads);

  void _apply_INV(ElementBase<> &gate);
  void _apply_AND(ElementBase<> &gate);
  void _apply_OR(ElementBase<> &gate);
  void _apply_XOR(ElementBase<> &gate);
  void _apply_NAND(ElementBase<> &gate);
  void _apply_NOR(ElementBase<> &gate);
  void _apply_XNOR(ElementBase<> &gate);
  void _apply_MUX(ElementBase<> &gate);
  void _apply_CLKBUF(ElementBase<> &gate);
  void _apply_PI(ElementBase<> &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(ElementBase<> &gate);
  void _run_gate(ElementBase<> &gate, const Pattern pattern, const size_t pi,
                 const size_t SA_fault, const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results, const size_t bits,
                                const size_t num_shift_gates);


  // Print for check 
  std::string gateTypeToString(GateType type) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                      const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_ppg(const ElementBase<> &gate) const;
  void print_simulation_results(const std::vector<size_t> &pi_results,
                                const std::vector<size_t> &gate_results,
                                const std::vector<size_t> &po_results) const;
  void _print_find_clusters() const;
  void print_hyperedge_vertices_details() const;
  void print_preconstruct_hyperedge_vertices_details() const;
  void _print_RepCut_output_cones();

  // Check 
  void _print_num_PIs_Gates_POs(Mode mode);
  void _traverse_cones_for_print(Mode mode, std::vector<size_t> &ret);
  void _count_duplications(Mode mode);

  double _round_to(double value, double precision = 1.0){
    return std::round(value / precision) * precision;
  }
};

#endif