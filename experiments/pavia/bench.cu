#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int pixels = 1600, bands = 103, threads = 256, warps = threads / 32;
constexpr int pairs = pixels * (pixels - 1) / 2, blocks = (pairs + warps - 1) / warps;
using Word = uint16_t;
using Score = unsigned long long;

#define CUDA(expr) do { auto status = (expr); if (status != cudaSuccess) \
    throw std::runtime_error(cudaGetErrorString(status)); } while (0)

template <class T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t count) { CUDA(cudaMalloc(&ptr, count * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
};

__host__ __device__ int triangle(int row) { return row * (row - 1) / 2; }

__device__ Word read_band(const Word* address) {
    Word value;
    asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(value) : "l"(address));
    return value;
}

extern "C" __global__ void spectral_pair(const Word* image, int pitch, Score* out) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int id = blockIdx.x * warps + warp;
    __shared__ Score scores[warps];
    Score sum = 0;
    if (id < pairs) {
        int row = int((1.0f + sqrtf(1.0f + 8.0f * id)) * 0.5f);
        if (triangle(row) > id) --row;
        if (triangle(row + 1) <= id) ++row;
        const int col = id - triangle(row);
        const Word* a = image + row * pitch;
        const Word* b = image + col * pitch;
        // Each iteration reads a consecutive window from each of two spectra.
        #pragma unroll
        for (int start = 0; start < bands; start += 32) {
            if (start + lane < bands) {
                const long long delta = int(read_band(a + start + lane)) -
                                        int(read_band(b + start + lane));
                sum += delta * delta;
            }
        }
        for (int offset = 16; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) scores[warp] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        Score best = 0;
        for (int i = 0; i < warps; ++i) {
            const int pair = blockIdx.x * warps + i;
            if (pair < pairs) {
                const Score result = (scores[i] << 21) | (pairs - pair);
                if (result > best) best = result;
            }
        }
        out[blockIdx.x] = best;
    }
}

__global__ void condition(const unsigned* data, size_t words, unsigned* out) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned sum = 0;
    for (size_t i = tid; i < words; i += gridDim.x * blockDim.x) {
        unsigned value;
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(data + i));
        sum += value;
    }
    out[tid] = sum;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("bench input.u16 PAD");
        const int pad = std::stoi(argv[2]), pitch = bands + pad;
        if (pad != 0 && pad != 9) throw std::runtime_error("Unexpected padding");
        std::vector<Word> data(pixels * bands), layout(pixels * pitch);
        std::ifstream input(argv[1], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(Word)) ||
            input.peek() != EOF) throw std::runtime_error("Unexpected input size");
        for (int i = 0; i < pixels; ++i)
            std::copy_n(data.data() + i * bands, bands, layout.data() + i * pitch);
        cudaDeviceProp device{};
        CUDA(cudaGetDeviceProperties(&device, 0));
        const size_t scrub = size_t(device.l2CacheSize) * 4;
        const size_t peak = scrub + layout.size() * sizeof(Word) +
                            blocks * sizeof(Score) + 256 * threads * sizeof(unsigned);
        if (!scrub || peak > (256ull << 20)) throw std::runtime_error("Device budget exceeded");
        Buffer<Word> spectrum(layout.size());
        if (reinterpret_cast<uintptr_t>(spectrum.ptr) % 32) throw std::runtime_error("Unaligned input");
        Buffer<Score> output(blocks);
        Buffer<unsigned> stream(scrub / sizeof(unsigned)), sink(256 * threads);
        CUDA(cudaMemcpy(spectrum.ptr, layout.data(), layout.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CUDA(cudaMemset(stream.ptr, 0x5a, scrub));
        CUDA(cudaMemset(output.ptr, 0, blocks * sizeof(Score)));
        cudaEvent_t begin, end;
        CUDA(cudaEventCreate(&begin));
        CUDA(cudaEventCreate(&end));
        condition<<<256, threads>>>(stream.ptr, scrub / sizeof(unsigned), sink.ptr);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(begin));
        spectral_pair<<<blocks, threads>>>(spectrum.ptr, pitch, output.ptr);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(end));
        CUDA(cudaEventSynchronize(end));
        float ms = 0;
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        std::vector<Score> results(blocks);
        CUDA(cudaMemcpy(results.data(), output.ptr, blocks * sizeof(Score), cudaMemcpyDeviceToHost));
        const Score best = *std::max_element(results.begin(), results.end());
        CUDA(cudaEventDestroy(begin));
        CUDA(cudaEventDestroy(end));
        std::printf("{\"ms\":%.9g,\"peak_bytes\":%zu,\"score\":%llu,\"pair\":%llu}\n",
                    ms, peak, best >> 21, pairs - (best & ((1u << 21) - 1)));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
