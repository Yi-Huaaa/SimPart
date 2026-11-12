#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simpart {

namespace detail {
inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error '") + cudaGetErrorString(err) + "' in " +
                             expr + " at " + file + ":" + std::to_string(line));
  }
}
}  // namespace detail

#define SIMPART_CUDA_CHECK(expr) ::simpart::detail::cuda_check((expr), #expr, __FILE__, __LINE__)

// Checks for errors of the most recent kernel launch.
#define SIMPART_CUDA_CHECK_LAUNCH() SIMPART_CUDA_CHECK(cudaGetLastError())

// Owning, move-only array in device memory.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) { resize(n); }
  explicit DeviceBuffer(const std::vector<T>& host) { upload(host); }
  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = std::exchange(other.ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  // Reallocates to exactly n elements (contents are undefined).
  void resize(size_t n) {
    reset();
    if (n > 0) SIMPART_CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    size_ = n;
  }

  void upload(const T* host, size_t n) {
    if (n != size_) resize(n);
    if (n > 0) SIMPART_CUDA_CHECK(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void upload(const std::vector<T>& host) { upload(host.data(), host.size()); }

  void download(T* host, size_t n) const {
    if (n > 0) SIMPART_CUDA_CHECK(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
  std::vector<T> download() const {
    std::vector<T> host(size_);
    download(host.data(), size_);
    return host;
  }

  void reset() noexcept {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
  }

  T* data() { return ptr_; }
  const T* data() const { return ptr_; }
  size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

}  // namespace simpart
