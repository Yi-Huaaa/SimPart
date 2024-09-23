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


// Forward declaration
class CUDASimulator;

// constexpr
constexpr auto SIZE_T_BITS = sizeof(size_t) * CHAR_BIT;

class CPUPartitioner {
  
  // Declare friend class 
  friend class CUDASimulator;

public:
  enum class Mode { SEQUENTIAL = 0, PARALLEL_OMP, CONES_SEQUENTIAL, CONES_PARALLEL_OMP };

  // accessor
  void read(const std::string &ckt, const std::string &flst, const std::string &ptn);
  void read(std::istream &ckt, std::istream &flst, std::istream &ptn);
  void run(Mode mode, const size_t num_threads);
  void visualization_graph(const std::string &outputFile);
  

private:
  // Basic private members 
  size_t _num_PIs, _num_POs, _num_inner_gates, _num_wires;
  size_t _sum_pi_gates_pos;
  std::vector<PI> _PIs;
  std::vector<Gate> _gates;
  std::vector<PO> _POs;
  size_t *_gate_input_order_table;
  
  // Fault and Pattern 
  size_t _num_fault; // for read file
  std::vector<Fault> _faults;

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
  size_t _pi_idx_accum, _po_idx_accum, _gate_idx_accum;

  // For merge
  // const size_t _merge_threshold = 11; // adjustable  
  // const size_t _merge_bound = 2;

  // Construct
  void _read_graph(std::istream &ckt, 
                  std::vector<std::vector<size_t>> &adj,
                  std::vector<std::vector<size_t>> &invAdj);                  
  void _levelization(std::vector<std::vector<size_t>> &adj);
  

  void _partition(std::vector<std::vector<size_t>> &adj);
  void _show_cones_size();
  bool _gateHasCone(std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    const size_t gateIdx);
  bool _gateHasPostGates(std::vector<std::vector<size_t>> &adj, 
                        const size_t gateIdx, const size_t startLevel);         
  void _getConeIdices(const size_t conesSz, const size_t gateIdx, const size_t startLevel, 
                      bool newCone, 
                      std::vector<std::vector<size_t>> &adj, 
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones);
  void _updateConIdxTable(bool newCone, const size_t gateIdx, 
                          std::vector<size_t> &coneIdxToSinkGateIdx);
  void _update_cones(bool newCone, const size_t gateIdx, 
                    const size_t startLevel, 
                    std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                    size_t &coneWidth);
  void _find_all_cones(const size_t startLevel, 
                          std::vector<std::vector<size_t>> &adj, 
                          std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                          std::vector<size_t> &coneIdxToSinkGateIdx,
                          int &stopLevel);
void _construct_duplication_table(std::vector<size_t> &coneIdxToSinkGateIdx,
                      std::vector<std::vector<size_t>> &gateBelongsToWhichCones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable,
                      const size_t startLevel, const int stopLevel, 
                      const size_t numLevels);

  void _merge_cones(std::vector<std::vector<std::vector<size_t>>> &cones,
                      std::vector<std::vector<std::vector<size_t>>> &duplicationTable,
                      std::vector<std::vector<std::vector<size_t>>> &outputCones);
  
  void _construct_graph(std::istream &ckt);
  void _read_pattern(std::istream &ptn);
  void _read_fault(std::istream &flst);

  // Simulation - sequentail and parallel 
  void _run_gates(const size_t num_threads);
  void _run_good_case(const Pattern pattern, const size_t bits, const size_t num_threads);
  void _run_bad_case(const Fault &fault, const Pattern pattern, 
                      const size_t bits, const size_t num_threads);

  void _apply_INV(ElementBase &gate);
  void _apply_AND(ElementBase &gate);
  void _apply_OR(ElementBase &gate);
  void _apply_XOR(ElementBase &gate);
  void _apply_NAND(ElementBase &gate);
  void _apply_NOR(ElementBase &gate);
  void _apply_XNOR(ElementBase &gate);
  void _apply_MUX(ElementBase &gate);
  void _apply_CLKBUF(ElementBase &gate);
  void _apply_PI(ElementBase &gate, const Pattern pattern, const size_t pi);
  void _apply_PO(ElementBase &gate);
  void _run_gate(ElementBase &gate, const Pattern pattern, const size_t pi,
                 const size_t SA_fault, const size_t fault_val);
  void _shift_to_correct_answer(std::vector<size_t> &results, const size_t bits,
                                const size_t num_shift_gates);


  // Print for check 
  std::string gateTypeToString(GateType type) const;
  void print_patterns(const std::vector<Pattern> &patterns, const size_t round,
                      const size_t num_PIs) const;
  void print_bits_stack(const size_t size, const void *const ptr) const;
  void print_ppg(const ElementBase &gate) const;
  void print_simulation_results(const std::vector<size_t> &pi_results,
                                const std::vector<size_t> &gate_results,
                                const std::vector<size_t> &po_results) const;
};

#endif