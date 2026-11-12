#include "simpart/gpu_simulator.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "simpart/cuda_utils.hpp"
#include "simpart/gate_eval.hpp"
#include "simpart/timer.hpp"

namespace simpart {

const char* to_string(Executor executor) {
  return executor == Executor::Stream ? "stream" : "graph";
}

namespace {

// One slot of the simulation schedule: a node and its fanins, stored in launch
// order so that a thread reads everything it needs with one 16-byte load.
struct alignas(16) NodeRecord {
  int32_t node;
  uint8_t type;
  uint8_t unused;
  uint16_t num_fanins;
  int32_t fanin0;  // first fanin, or offset into DeviceCircuit::fanins if num_fanins > 2
  int32_t fanin1;  // second fanin (== fanin0 if num_fanins == 1)
};
static_assert(sizeof(NodeRecord) == 16, "NodeRecord must be 16 bytes");

std::vector<NodeRecord> make_records(const Circuit& c, const std::vector<int32_t>& nodes) {
  std::vector<NodeRecord> records(nodes.size());
  for (size_t i = 0; i < nodes.size(); ++i) {
    const int32_t v = nodes[i];
    const int32_t n = c.fanin_count(v);
    const int32_t begin = c.fanin_offsets[v];
    if (n > UINT16_MAX) throw std::runtime_error("node with more than 65535 fanins");
    NodeRecord& r = records[i];
    r = {v, static_cast<uint8_t>(c.types[v]), 0, static_cast<uint16_t>(n), v, v};
    if (n > 2) {
      r.fanin0 = begin;
    } else if (n > 0) {
      r.fanin0 = c.fanins[begin];
      r.fanin1 = c.fanins[begin + n - 1];
    }
  }
  return records;
}

// Device-side view of the circuit and the simulation state. A pass simulates
// `batch` consecutive rounds; every node holds one 64-bit word per round of the
// pass.
struct DeviceCircuit {
  const int32_t* fanins;     // CSR fanins, for nodes with more than two fanins
  const uint64_t* patterns;  // [(pass * num_pis + pi) * batch + word]
  uint64_t* values;          // [node * batch + word]
  uint64_t* po_values;       // [round * num_pos + po]
  int32_t num_pis;
  int32_t num_pos;
  int32_t num_rounds;
  int32_t batch;
};

__device__ __forceinline__ NodeRecord load_record(const NodeRecord* p) {
  const int4 raw = __ldg(reinterpret_cast<const int4*>(p));
  NodeRecord r;
  r.node = raw.x;
  r.type = static_cast<uint8_t>(raw.y & 0xff);
  r.num_fanins = static_cast<uint16_t>(static_cast<uint32_t>(raw.y) >> 16);
  r.fanin0 = raw.z;
  r.fanin1 = raw.w;
  return r;
}

// Loads/stores V consecutive words (V = 1, 2 or 4) with 16-byte accesses.
template <int V>
__device__ __forceinline__ void load_words(const uint64_t* p, uint64_t (&v)[V]) {
  if constexpr (V == 1) {
    v[0] = p[0];
  } else {
#pragma unroll
    for (int i = 0; i < V; i += 2) {
      const ulonglong2 x = *reinterpret_cast<const ulonglong2*>(p + i);
      v[i] = x.x;
      v[i + 1] = x.y;
    }
  }
}

template <int V>
__device__ __forceinline__ void store_words(uint64_t* p, const uint64_t (&v)[V]) {
  if constexpr (V == 1) {
    p[0] = v[0];
  } else {
#pragma unroll
    for (int i = 0; i < V; i += 2) {
      *reinterpret_cast<ulonglong2*>(p + i) = make_ulonglong2(v[i], v[i + 1]);
    }
  }
}

// Evaluates one schedule slot for all rounds of the pass stored at *pass, V
// words at a time. B is the batch size if it is known at compile time (B == V),
// or 0 to read it from the circuit. Replicated nodes are written by several
// blocks, always with the same values.
template <int V, int B>
__device__ __forceinline__ void eval_record(const NodeRecord& r, const DeviceCircuit& c,
                                            const int32_t* pass) {
  const GateType type = static_cast<GateType>(r.type);
  const int32_t batch = B > 0 ? B : c.batch;
  for (int32_t w0 = 0; w0 < batch; w0 += V) {
    uint64_t out[V];
    if (type == GateType::PI) {
      load_words<V>(c.patterns + (int64_t(*pass) * c.num_pis + r.node) * batch + w0, out);
    } else if (r.num_fanins <= 2) {
      uint64_t a[V], b[V];
      load_words<V>(c.values + int64_t(r.fanin0) * batch + w0, a);
      load_words<V>(c.values + int64_t(r.fanin1) * batch + w0, b);
#pragma unroll
      for (int w = 0; w < V; ++w) {
        out[w] = eval_gate(type, r.num_fanins, [&](int32_t i) { return i == 0 ? a[w] : b[w]; });
      }
    } else {
      const int32_t* fanin = c.fanins + r.fanin0;
#pragma unroll
      for (int w = 0; w < V; ++w) {
        out[w] = eval_gate(type, r.num_fanins,
                           [&](int32_t i) { return c.values[int64_t(fanin[i]) * batch + w0 + w]; });
      }
    }
    if (type == GateType::PO) {
#pragma unroll
      for (int w = 0; w < V; ++w) {
        const int32_t round = *pass * batch + w0 + w;
        if (round < c.num_rounds) {
          c.po_values[int64_t(round) * c.num_pos + (r.node - c.num_pis)] = out[w];
        }
      }
    }
    store_words<V>(c.values + int64_t(r.node) * batch + w0, out);
  }
}

// Algorithm 3: one DSP level, one thread per node (inter-block sync = kernel boundary).
template <int V, int B>
__global__ void __launch_bounds__(GpuSimulator::kBlockSize)
    dsp_level_kernel(const NodeRecord* __restrict__ records, int32_t count, DeviceCircuit c,
                     const int32_t* pass) {
  const int32_t t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < count) eval_record<V, B>(load_record(records + t), c, pass);
}

// Algorithm 4: one block per group walks the RAP levels of its cone; levels
// wider than the block take several passes; intra-block sync between levels.
// The level offsets of the group are first copied to shared memory.
template <int V, int B>
__global__ void __launch_bounds__(GpuSimulator::kBlockSize)
    rap_kernel(const int32_t* __restrict__ offsets, const NodeRecord* __restrict__ records,
               int32_t rap_levels, DeviceCircuit c, const int32_t* pass) {
  extern __shared__ int32_t level_offsets[];
  const int32_t* group_offsets = offsets + int64_t(blockIdx.x) * rap_levels;
  for (int32_t l = threadIdx.x; l <= rap_levels; l += blockDim.x) level_offsets[l] = group_offsets[l];
  __syncthreads();
  for (int32_t l = 0; l < rap_levels; ++l) {
    const int32_t end = level_offsets[l + 1];
    for (int32_t i = level_offsets[l] + threadIdx.x; i < end; i += blockDim.x) {
      eval_record<V, B>(load_record(records + i), c, pass);
    }
    __syncthreads();
  }
}

// Loop control of the conditional graph: moves to the next pass and stops the
// while loop after the last one.
__global__ void advance_pass_kernel(int32_t* pass, int32_t num_passes,
                                    cudaGraphConditionalHandle handle) {
  const int32_t next = *pass + 1;
  *pass = next;
  if (next >= num_passes) cudaGraphSetConditional(handle, 0);
}

}  // namespace

struct GpuSimulator::Impl {
  int32_t num_pis = 0, num_pos = 0, num_rounds = 0;
  int32_t batch = 1, num_passes = 0;
  int32_t dsp_levels = 0, rap_levels = 0, num_groups = 0;
  std::vector<int32_t> dsp_offsets;  // host copy, for launch configurations

  DeviceBuffer<int32_t> fanins;
  DeviceBuffer<uint64_t> patterns, values, po_values;
  DeviceBuffer<NodeRecord> dsp_records, rap_records;
  DeviceBuffer<int32_t> rap_offsets;
  DeviceBuffer<int32_t> pass_ids;      // 0, 1, ..., num_passes-1 (stream executor)
  DeviceBuffer<int32_t> pass_counter;  // current pass (graph executor)

  cudaStream_t stream = nullptr;
  cudaGraphExec_t graph_exec = nullptr;

  DeviceCircuit device_circuit() {
    return {fanins.data(), patterns.data(), values.data(), po_values.data(),
            num_pis,       num_pos,         num_rounds,    batch};
  }

  // Enqueues the kernels that simulate one pass, reading the pass index at *pass.
  template <int V, int B>
  void enqueue_pass(cudaStream_t s, const int32_t* pass) {
    const DeviceCircuit c = device_circuit();
    for (int32_t l = 0; l < dsp_levels; ++l) {
      const int32_t count = dsp_offsets[l + 1] - dsp_offsets[l];
      if (count == 0) continue;
      const int32_t threads = std::min(count, kDspBlockSize);
      const int32_t blocks = (count + kDspBlockSize - 1) / kDspBlockSize;
      dsp_level_kernel<V, B><<<blocks, threads, 0, s>>>(dsp_records.data() + dsp_offsets[l], count, c,
                                                     pass);
    }
    if (rap_levels > 0 && num_groups > 0) {
      const size_t shared_bytes = (static_cast<size_t>(rap_levels) + 1) * sizeof(int32_t);
      rap_kernel<V, B><<<num_groups, kBlockSize, shared_bytes, s>>>(rap_offsets.data(),
                                                                 rap_records.data(), rap_levels, c,
                                                                 pass);
    }
    SIMPART_CUDA_CHECK_LAUNCH();
  }

  void enqueue_pass(cudaStream_t s, const int32_t* pass) {
    if (batch == 1) return enqueue_pass<1, 1>(s, pass);
    if (batch == 2) return enqueue_pass<2, 2>(s, pass);
    if (batch == 4) return enqueue_pass<4, 4>(s, pass);
    return enqueue_pass<4, 0>(s, pass);
  }
};

GpuSimulator::GpuSimulator(const Circuit& circuit, const Partition& partition,
                           const Patterns& patterns, int32_t batch)
    : impl_(std::make_unique<Impl>()) {
  if (patterns.num_pis != circuit.num_pis) {
    throw std::invalid_argument("patterns do not match the number of PIs of the circuit");
  }
  if (batch < 1 || batch > kMaxBatch || (batch & (batch - 1)) != 0) {
    throw std::invalid_argument("batch must be 1, 2, 4, 8 or 16");
  }
  Impl& s = *impl_;
  s.num_pis = circuit.num_pis;
  s.num_pos = circuit.num_pos;
  s.num_rounds = patterns.num_rounds;
  s.batch = batch;
  s.num_passes = (s.num_rounds + batch - 1) / batch;
  s.dsp_levels = partition.dsp_levels;
  s.rap_levels = partition.rap_levels();
  s.num_groups = partition.num_groups;
  s.dsp_offsets = partition.dsp_offsets;

  // Patterns grouped by pass: the words of one PI for all rounds of a pass are
  // contiguous. A last partial pass is padded with zero patterns.
  std::vector<uint64_t> words(static_cast<size_t>(s.num_passes) * s.num_pis * batch, 0);
  for (int32_t r = 0; r < s.num_rounds; ++r) {
    const int32_t pass = r / batch, w = r % batch;
    for (int32_t pi = 0; pi < s.num_pis; ++pi) {
      words[(static_cast<size_t>(pass) * s.num_pis + pi) * batch + w] =
          patterns.words[static_cast<size_t>(r) * s.num_pis + pi];
    }
  }
  s.patterns.upload(words);
  s.fanins.upload(circuit.fanins);
  s.values.resize(static_cast<size_t>(circuit.num_nodes()) * batch);
  s.po_values.resize(static_cast<size_t>(s.num_rounds) * s.num_pos);
  if ((static_cast<size_t>(s.rap_levels) + 1) * sizeof(int32_t) > 48 * 1024) {
    throw std::runtime_error("too many RAP levels for the shared-memory level table");
  }
  s.dsp_records.upload(make_records(circuit, partition.dsp_nodes));
  s.rap_records.upload(make_records(circuit, partition.rap_nodes));
  s.rap_offsets.upload(partition.rap_offsets);

  std::vector<int32_t> ids(std::max(s.num_passes, 1));
  std::iota(ids.begin(), ids.end(), 0);
  s.pass_ids.upload(ids);
  s.pass_counter.resize(1);

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

  // pass = 0, then: while (cond) { simulate pass; ++pass; cond = pass < num_passes; }
  cudaMemsetParams reset{};
  reset.dst = s.pass_counter.data();
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
  s.enqueue_pass(s.stream, s.pass_counter.data());
  advance_pass_kernel<<<1, 1, 0, s.stream>>>(s.pass_counter.data(), s.num_passes, handle);
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
    for (int32_t p = 0; p < s.num_passes; ++p) s.enqueue_pass(s.stream, s.pass_ids.data() + p);
  }
  SIMPART_CUDA_CHECK(cudaStreamSynchronize(s.stream));
}

std::vector<uint64_t> GpuSimulator::po_values() const { return impl_->po_values.download(); }

std::vector<uint64_t> GpuSimulator::node_values() const {
  const Impl& s = *impl_;
  const std::vector<uint64_t> all = s.values.download();
  if (s.batch == 1 || s.num_rounds == 0) return all;
  // The values of the last round sit in word (num_rounds - 1) % batch of the last pass.
  const int32_t word = (s.num_rounds - 1) - (s.num_passes - 1) * s.batch;
  std::vector<uint64_t> last(all.size() / s.batch);
  for (size_t v = 0; v < last.size(); ++v) last[v] = all[v * s.batch + word];
  return last;
}

}  // namespace simpart
