#include "simpart/gpu_simulator.hpp"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

#include "simpart/cuda_utils.hpp"
#include "simpart/gate_eval.hpp"
#include "simpart/timer.hpp"

namespace simpart {

const char* to_string(Executor executor) {
  return executor == Executor::Stream ? "stream" : "graph";
}

namespace {

// Device-side view of the circuit and the simulation state.
struct DeviceCircuit {
  const GateType* types;
  const int32_t* fanin_offsets;
  const int32_t* fanins;
  const uint64_t* patterns;  // [round * num_pis + pi]
  uint64_t* values;          // [node]
  uint64_t* po_values;       // [round * num_pos + po]
  int32_t num_pis;
  int32_t num_pos;
};

struct LoadValue {
  const uint64_t* values;
  SIMPART_HOST_DEVICE uint64_t operator()(int32_t u) const { return values[u]; }
};

// Evaluates node v for the round stored at *round. Replicated nodes are
// written by several blocks, always with the same value.
__device__ __forceinline__ void eval_node(int32_t v, const DeviceCircuit& c, const int32_t* round) {
  const GateType type = c.types[v];
  uint64_t value;
  if (type == GateType::PI) {
    value = c.patterns[int64_t(*round) * c.num_pis + v];
  } else {
    const int32_t begin = c.fanin_offsets[v];
    value = eval_gate(type, c.fanins + begin, c.fanin_offsets[v + 1] - begin, LoadValue{c.values});
    if (type == GateType::PO) c.po_values[int64_t(*round) * c.num_pos + (v - c.num_pis)] = value;
  }
  c.values[v] = value;
}

// Algorithm 3: one DSP level, one thread per node (inter-block sync = kernel boundary).
__global__ void __launch_bounds__(GpuSimulator::kBlockSize)
    dsp_level_kernel(const int32_t* __restrict__ nodes, int32_t count, DeviceCircuit c,
                     const int32_t* round) {
  const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < count) eval_node(nodes[t], c, round);
}

// [experimental] All DSP levels in one cooperative kernel, with a grid-wide
// barrier between levels instead of a kernel boundary.
__global__ void __launch_bounds__(GpuSimulator::kBlockSize)
    dsp_cooperative_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ nodes,
                           int32_t dsp_levels, DeviceCircuit c, const int32_t* round) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int32_t stride = gridDim.x * blockDim.x;
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t l = 0; l < dsp_levels; ++l) {
    const int32_t end = offsets[l + 1];
    for (int32_t i = offsets[l] + tid; i < end; i += stride) eval_node(nodes[i], c, round);
    grid.sync();
  }
}

// Algorithm 4: one block per group walks the RAP levels of its cone; levels
// wider than the block take several passes; intra-block sync between levels.
__global__ void __launch_bounds__(GpuSimulator::kBlockSize)
    rap_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ nodes,
               int32_t rap_levels, DeviceCircuit c, const int32_t* round) {
  const int32_t* group_offsets = offsets + int64_t(blockIdx.x) * rap_levels;
  for (int32_t l = 0; l < rap_levels; ++l) {
    const int32_t end = group_offsets[l + 1];
    for (int32_t i = group_offsets[l] + threadIdx.x; i < end; i += blockDim.x) {
      eval_node(nodes[i], c, round);
    }
    __syncthreads();
  }
}

// Loop control of the conditional graph: moves to the next round and stops
// the while loop after the last one.
__global__ void advance_round_kernel(int32_t* round, int32_t num_rounds,
                                     cudaGraphConditionalHandle handle) {
  const int32_t next = *round + 1;
  *round = next;
  if (next >= num_rounds) cudaGraphSetConditional(handle, 0);
}

}  // namespace

struct GpuSimulator::Impl {
  int32_t num_pis = 0, num_pos = 0, num_rounds = 0;
  int32_t dsp_levels = 0, rap_levels = 0, num_groups = 0;
  std::vector<int32_t> dsp_offsets;  // host copy, for launch configurations

  DeviceBuffer<GateType> types;
  DeviceBuffer<int32_t> fanin_offsets, fanins;
  DeviceBuffer<uint64_t> patterns, values, po_values;
  DeviceBuffer<int32_t> dsp_nodes, rap_offsets, rap_nodes;
  DeviceBuffer<int32_t> dsp_offsets_device;
  int cooperative_blocks = 0;  // > 0: run the DSP region with dsp_cooperative_kernel
  DeviceBuffer<int32_t> round_ids;      // 0, 1, ..., num_rounds-1 (stream executor)
  DeviceBuffer<int32_t> round_counter;  // current round (graph executor)

  cudaStream_t stream = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  DeviceCircuit device_circuit() {
    return {types.data(),    fanin_offsets.data(), fanins.data(), patterns.data(),
            values.data(),   po_values.data(),     num_pis,       num_pos};
  }

  // Enqueues the kernels that simulate one round, reading the round at *round.
  void enqueue_round(cudaStream_t s, const int32_t* round) {
    const DeviceCircuit c = device_circuit();
    if (cooperative_blocks > 0 && dsp_levels > 0) {
      cudaLaunchConfig_t config{};
      config.gridDim = dim3(cooperative_blocks);
      config.blockDim = dim3(kBlockSize);
      config.stream = s;
      cudaLaunchAttribute attribute{};
      attribute.id = cudaLaunchAttributeCooperative;
      attribute.val.cooperative = 1;
      config.attrs = &attribute;
      config.numAttrs = 1;
      SIMPART_CUDA_CHECK(cudaLaunchKernelEx(&config, dsp_cooperative_kernel,
                                            dsp_offsets_device.data(), dsp_nodes.data(),
                                            dsp_levels, c, round));
    }
    for (int32_t l = 0; cooperative_blocks == 0 && l < dsp_levels; ++l) {
      const int32_t count = dsp_offsets[l + 1] - dsp_offsets[l];
      if (count == 0) continue;
      const int32_t threads = std::min(count, kBlockSize);
      const int32_t blocks = (count + kBlockSize - 1) / kBlockSize;
      dsp_level_kernel<<<blocks, threads, 0, s>>>(dsp_nodes.data() + dsp_offsets[l], count, c,
                                                  round);
    }
    if (rap_levels > 0 && num_groups > 0) {
      rap_kernel<<<num_groups, kBlockSize, 0, s>>>(rap_offsets.data(), rap_nodes.data(),
                                                   rap_levels, c, round);
    }
    SIMPART_CUDA_CHECK_LAUNCH();
  }
};

GpuSimulator::GpuSimulator(const Circuit& circuit, const Partition& partition,
                           const Patterns& patterns)
    : impl_(std::make_unique<Impl>()) {
  if (patterns.num_pis != circuit.num_pis) {
    throw std::invalid_argument("patterns do not match the number of PIs of the circuit");
  }
  Impl& s = *impl_;
  s.num_pis = circuit.num_pis;
  s.num_pos = circuit.num_pos;
  s.num_rounds = patterns.num_rounds;
  s.dsp_levels = partition.dsp_levels;
  s.rap_levels = partition.rap_levels();
  s.num_groups = partition.num_groups;
  s.dsp_offsets = partition.dsp_offsets;

  s.types.upload(circuit.types);
  s.fanin_offsets.upload(circuit.fanin_offsets);
  s.fanins.upload(circuit.fanins);
  s.patterns.upload(patterns.words);
  s.values.resize(circuit.num_nodes());
  s.po_values.resize(static_cast<size_t>(s.num_rounds) * s.num_pos);
  s.dsp_nodes.upload(partition.dsp_nodes);
  s.rap_offsets.upload(partition.rap_offsets);
  s.rap_nodes.upload(partition.rap_nodes);
  if (const char* env = std::getenv("SIMPART_COOPERATIVE_DSP"); env != nullptr && env[0] == '1') {
    int device = 0, sms = 0, per_sm = 0;
    SIMPART_CUDA_CHECK(cudaGetDevice(&device));
    SIMPART_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    SIMPART_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &per_sm, dsp_cooperative_kernel, kBlockSize, 0));
    s.cooperative_blocks = sms * per_sm;
    s.dsp_offsets_device.upload(partition.dsp_offsets);
  }

  std::vector<int32_t> ids(std::max(s.num_rounds, 1));
  std::iota(ids.begin(), ids.end(), 0);
  s.round_ids.upload(ids);
  s.round_counter.resize(1);

  if (s.values.size() > 0) {
    SIMPART_CUDA_CHECK(cudaMemset(s.values.data(), 0, s.values.size() * sizeof(uint64_t)));
  }
  SIMPART_CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
}

GpuSimulator::~GpuSimulator() {
  if (impl_->graph_exec != nullptr) cudaGraphExecDestroy(impl_->graph_exec);
  if (impl_->stream != nullptr) cudaStreamDestroy(impl_->stream);
}

double GpuSimulator::prepare_graph() {
  Impl& s = *impl_;
  if (s.graph_exec != nullptr) return 0.0;
  Timer timer;

  cudaGraph_t graph = nullptr;
  SIMPART_CUDA_CHECK(cudaGraphCreate(&graph, 0));
  cudaGraphConditionalHandle handle;
  SIMPART_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault));

  // round = 0, then: while (cond) { simulate round; ++round; cond = round < num_rounds; }
  cudaMemsetParams reset{};
  reset.dst = s.round_counter.data();
  reset.value = 0;
  reset.elementSize = sizeof(int32_t);
  reset.width = 1;
  reset.height = 1;
  reset.pitch = sizeof(int32_t);
  cudaGraphNode_t reset_node;
  SIMPART_CUDA_CHECK(cudaGraphAddMemsetNode(&reset_node, graph, nullptr, 0, &reset));

  cudaGraphNodeParams loop{};
  loop.type = cudaGraphNodeTypeConditional;
  loop.conditional.handle = handle;
  loop.conditional.type = cudaGraphCondTypeWhile;
  loop.conditional.size = 1;
  cudaGraphNode_t loop_node;
  SIMPART_CUDA_CHECK(cudaGraphAddNode(&loop_node, graph, &reset_node, 1, &loop));

  // The loop body is captured from the same launch code as the stream executor,
  // so its kernels form a chain in launch order.
  cudaGraph_t body = loop.conditional.phGraph_out[0];
  SIMPART_CUDA_CHECK(cudaStreamBeginCaptureToGraph(s.stream, body, nullptr, nullptr, 0,
                                                   cudaStreamCaptureModeThreadLocal));
  s.enqueue_round(s.stream, s.round_counter.data());
  advance_round_kernel<<<1, 1, 0, s.stream>>>(s.round_counter.data(), s.num_rounds, handle);
  SIMPART_CUDA_CHECK_LAUNCH();
  SIMPART_CUDA_CHECK(cudaStreamEndCapture(s.stream, &body));

  SIMPART_CUDA_CHECK(cudaGraphInstantiate(&s.graph_exec, graph, 0));
  SIMPART_CUDA_CHECK(cudaGraphDestroy(graph));
  SIMPART_CUDA_CHECK(cudaGraphUpload(s.graph_exec, s.stream));
  SIMPART_CUDA_CHECK(cudaStreamSynchronize(s.stream));
  return timer.elapsed_ms();
}

void GpuSimulator::run(Executor executor) {
  Impl& s = *impl_;
  if (s.num_rounds == 0) return;
  if (executor == Executor::Graph) {
    prepare_graph();
    SIMPART_CUDA_CHECK(cudaGraphLaunch(s.graph_exec, s.stream));
  } else {
    for (int32_t r = 0; r < s.num_rounds; ++r) s.enqueue_round(s.stream, s.round_ids.data() + r);
  }
  SIMPART_CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

std::vector<uint64_t> GpuSimulator::po_values() const { return impl_->po_values.download(); }

std::vector<uint64_t> GpuSimulator::node_values() const { return impl_->values.download(); }

}  // namespace simpart
