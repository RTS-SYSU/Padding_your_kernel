// Independent L2 eviction and fill controls. Build with the same SM as bench.cu.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Word = unsigned;
using Cycles = unsigned long long;
constexpr int block_words = 32;
constexpr int buffer_count = 3;
constexpr int trials = 12;

inline void check(cudaError_t code, const char* expression) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(code));
}
#define CUDA(call) check((call), #call)

struct Buffer {
    void* raw = nullptr;
    explicit Buffer(size_t bytes) { CUDA(cudaMalloc(&raw, bytes)); }
    ~Buffer() { cudaFree(raw); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<typename T> T* as() const { return static_cast<T*>(raw); }
};

__device__ __forceinline__ Word load_cg(const Word* address) {
    Word value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
    return value;
}

// Every stored word has low bit zero. The potentially trapping instruction
// forces the consuming clock read to wait for the register scoreboard.
__device__ __forceinline__ Cycles clock_after(Word value) {
    Cycles timestamp;
    asm volatile("{ .reg .pred bad; setp.ne.u32 bad, %1, 0; @bad trap;"
                 " mov.u64 %0, %%clock64; }"
                 : "=l"(timestamp) : "r"(value & 1u) : "memory");
    return timestamp;
}

__device__ __forceinline__ Cycles timed_load(const Word* address, Word& previous) {
    const Cycles start = clock_after(previous);
    // This high bit remains zero for the lifetime of the experiment. It creates
    // a real address dependency from the timestamp to the timed load.
    previous = load_cg(address + (start >> 63));
    return clock_after(previous) - start;
}

__host__ __device__ Word initial_word(size_t index, Word seed) {
    Word value = Word(index) + seed;
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return (value ^ (value >> 16)) & ~1u;
}

__global__ void initialize(Word* data, size_t count, Word seed) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += gridDim.x * blockDim.x) {
        data[i] = initial_word(i, seed);
    }
}

__global__ void scrub_cache(const Word* data, size_t count, Word* sink) {
    const int thread = blockIdx.x * blockDim.x + threadIdx.x;
    Word sum = 0;
    for (size_t i = thread; i < count; i += gridDim.x * blockDim.x)
        sum += load_cg(data + i);
    sink[thread] = sum;
}

enum class Pattern { strided, neighbor, capacity };
struct Condition { Pattern pattern; int stride; int competitors; int sector_word; };
struct Sample { Cycles cold, hot, probe; };
struct Scheduled { int allocation, trial, base_block; Condition condition; };

enum class Pressure { empty, capacity, strided, jitter };
struct PressureCase { Pressure pattern; int stride, competitors; };
struct PressureSample { int allocation, trial, anchor; Word seed; PressureCase condition; };

// A candidate at each 1-based bin endpoint versus one candidate in each bin.
// Keeping both endpoint bins fixed gives both patterns exactly the same span.
__host__ __device__ Word pressure_offset(Word bin, int count, Word stride, Word seed, bool jitter) {
    Word position = stride;
    if (jitter && bin > 0 && bin + 1 < Word(count)) {
        Word value = bin ^ seed;
        value ^= value >> 16;
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        value ^= value >> 16;
        position = 1u + (value & (stride - 1));
    }
    return bin * stride + position;
}

__global__ void insert_pressure(const Word* base, int n, Word stride, Word seed, bool jitter) {
    Word sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        const Word offset = pressure_offset(Word(i), n, stride, seed, jitter);
        sum += load_cg(base + size_t(offset) * block_words);
    }
    // All initialized values are even. Consuming every sum keeps every load
    // observable without filling additional target-cache sectors with stores.
    if (sum & 1u) asm volatile("trap;" ::: "memory");
}

__global__ void measure(const Word* data, int anchor, Condition c, Sample* out,
                        bool post_capacity_scrub) {
    const Word* target = data + size_t(anchor) * block_words;
    // Same 4 KiB virtual page, distinct block: refresh the translation before
    // timing each access. This extra touched block is explicit in the protocol.
    const Word* witness = data + size_t(anchor ^ 8) * block_words;
    Word previous = load_cg(witness);
    Cycles cold = 0;
    if (!post_capacity_scrub) {
        cold = timed_load(target, previous);
        if (c.pattern == Pattern::capacity) {
            out->cold = cold;
            return;
        }
        for (int i = 1; i <= c.competitors; ++i)
            previous = load_cg(target + size_t(i) * c.stride * block_words + (previous & 1u));
    }
    previous = load_cg(witness + (previous & 1u));
    const Word* tested_sector = target + c.sector_word;
    const Cycles probe = timed_load(tested_sector, previous);
    previous = load_cg(witness + (previous & 1u));
    const Cycles hot = timed_load(tested_sector, previous);
    if (!post_capacity_scrub) out->cold = cold;
    out->probe = probe;
    out->hot = hot;
}

__global__ void refresh_witness(const Word* witness) {
    const Word first = load_cg(witness);
    if (load_cg(witness + (first & 1u)) & 1u) asm volatile("trap;" ::: "memory");
}

// This is the only kernel selected by NCU in --counter-check. One lane reads
// exactly one sector. Both earlier witness loads belong to an unprofiled launch.
__device__ __forceinline__ void record_anchor(const Word* address, Cycles* elapsed, Cycles* read_value) {
    Word observed = 0;
    const Cycles cycles = timed_load(address, observed);
    *elapsed = cycles;
    *read_value = observed;
}

extern "C" __global__ void probe_read_anchor(const Word* address, Cycles* elapsed, Cycles* read_value) {
    record_anchor(address, elapsed, read_value);
}

extern "C" __global__ void probe_native_anchor(const Word* address, Cycles* elapsed, Cycles* read_value) {
    record_anchor(address, elapsed, read_value);
}

bool native_first(const PressureSample& s) {
    return ((s.allocation + s.trial + int(s.condition.pattern)) % 2) == 0;
}

int run_pressure(const cudaDeviceProp& device, int capacity_blocks, const std::string& mode) {
    const bool matched = mode == "--matched-counter";
    const bool counter_check = matched || mode == "--counter-check";
    const int max_span = std::min(1 << 21, 8 * capacity_blocks);
    std::vector<int> depths = {std::max(1, capacity_blocks / 64),
                               std::max(1, capacity_blocks / 16),
                               std::max(1, capacity_blocks / 4),
                               std::max(1, capacity_blocks / 2),
                               capacity_blocks};
    // Include full-pressure controls using a single touched sector per block.
    for (int m : {2, 4, 8})
        if (capacity_blocks <= max_span / m) depths.push_back(capacity_blocks * m);
    std::sort(depths.begin(), depths.end());
    depths.erase(std::unique(depths.begin(), depths.end()), depths.end());
    std::vector<PressureCase> cases{{Pressure::empty, 0, 0}, {Pressure::capacity, 0, 0}};
    if (counter_check) {
        if (capacity_blocks > max_span / 2)
            throw std::runtime_error("Counter-check geometry exceeds the probe allocation");
        cases.push_back({Pressure::strided, 2, capacity_blocks});
        cases.push_back({Pressure::jitter, 2, capacity_blocks});
    } else {
        for (int n : depths) {
            for (int stride = 1; stride <= max_span / n; stride *= 2) {
                cases.push_back({Pressure::strided, stride, n});
                cases.push_back({Pressure::jitter, stride, n});
            }
        }
    }
    constexpr int anchors[] = {3, 74, 217, 465, 738, 1013};
    std::vector<PressureSample> schedule;
    for (int a = 0; a < buffer_count; ++a)
        for (int trial = 0; trial < 6; ++trial)
            for (const auto& c : cases) {
                // Matched pairs use identical anchors and seeds; allocation 2
                // tests a new allocation with the same six relative geometries.
                const Word seed = 913027u + 0x9e3779b9u * Word(trial + 1);
                schedule.push_back({a, trial, anchors[trial], seed, c});
            }
    std::mt19937 rng(913027);
    std::shuffle(schedule.begin(), schedule.end(), rng);
    const size_t words = (size_t(max_span) + 1024) * block_words;
    const size_t bytes = words * sizeof(Word) + 4096;
    Buffer first(bytes), second(bytes), third(bytes);
    Buffer* buffers[buffer_count] = {&first, &second, &third};
    Word* tables[buffer_count]{};
    const int blocks = device.multiProcessorCount * 4;
    constexpr int threads = 256;
    for (int a = 0; a < buffer_count; ++a) {
        const auto aligned = (reinterpret_cast<std::uintptr_t>(buffers[a]->raw) + 4095) & ~std::uintptr_t(4095);
        tables[a] = reinterpret_cast<Word*>(aligned);
        initialize<<<blocks, threads>>>(tables[a], words, 257u * (a + 1));
    }
    const size_t scrub_bytes = size_t(device.l2CacheSize) * 2;
    Buffer scrub(scrub_bytes), sink(blocks * threads * sizeof(Word));
    initialize<<<blocks, threads>>>(scrub.as<Word>(), scrub_bytes / sizeof(Word), 2039);
    CUDA(cudaGetLastError());
    std::printf("{\"kind\":\"info\",\"mode\":\"%s\",\"device\":\"%s\",\"l2_bytes\":%d,"
                "\"scrub_bytes\":%zu,\"block_bytes\":128,\"virtual_witness_block_xor\":8,"
                "\"max_span_blocks\":%d,\"pressure_threads\":%d,\"pressure_blocks\":%d,"
                "\"planned_samples\":%zu,\"seed\":913027}\n",
                matched ? "matched_counter" : (counter_check ? "counter_check" : "pressure"),
                device.name, device.l2CacheSize,
                scrub_bytes, max_span, threads, blocks, schedule.size());
    std::fflush(stdout);

    const int arms = matched ? 2 : 1;
    std::vector<Sample> samples(schedule.size() * arms);
    Buffer result(samples.size() * sizeof(Sample));
    const Condition split{Pattern::capacity, 0, 0, 0};
    auto flush_l2 = [&]() {
        scrub_cache<<<blocks, threads>>>(scrub.as<Word>(), scrub_bytes / sizeof(Word), sink.as<Word>());
    };
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& s = schedule[i];
        const auto& c = s.condition;
        Word* table = tables[s.allocation];
        Sample* output = result.as<Sample>() + i;
        for (int arm = 0; arm < arms; ++arm) {
            flush_l2();
            measure<<<1, 1>>>(table, s.anchor, split, output, false);
            const Word* target = table + size_t(s.anchor) * block_words;
            if (c.pattern == Pressure::capacity) {
                flush_l2();
            } else {
                insert_pressure<<<blocks, threads>>>(target, c.competitors, Word(c.stride),
                                                    s.seed, c.pattern == Pressure::jitter);
            }
            if (counter_check) {
                refresh_witness<<<1, 1>>>(table + size_t(s.anchor ^ 8) * block_words);
                // The two terminal kernels write after their timed read.
                if (matched && (arm == 0) == native_first(s)) {
                    Sample* native = output + schedule.size();
                    probe_native_anchor<<<1, 1>>>(target, &native->probe, &native->hot);
                } else {
                    probe_read_anchor<<<1, 1>>>(target, &output->probe, &output->hot);
                }
            } else {
                measure<<<1, 1>>>(table, s.anchor, split, output, true);
            }
            CUDA(cudaGetLastError());
        }
    }
    CUDA(cudaMemcpy(samples.data(), result.raw, samples.size() * sizeof(Sample), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < schedule.size(); ++i) {
        const auto& s = schedule[i];
        const auto& c = s.condition;
        const char* labels[] = {"pressure_empty", "pressure_capacity", "pressure_strided", "pressure_jitter"};
        const auto block = Cycles(reinterpret_cast<std::uintptr_t>(tables[s.allocation]) / 128);
        if (counter_check) {
            const Word expected = initial_word(size_t(s.anchor) * block_words, 257u * (s.allocation + 1));
            if (samples[i].hot != expected) throw std::runtime_error("Single-read output mismatch");
            std::printf("{\"kind\":\"counter_case\",\"ordinal\":%zu,\"pattern\":\"%s\","
                        "\"allocation\":%d,\"trial\":%d,\"allocation_base_block\":%llu,"
                        "\"base_block\":%d,\"sector_word\":0,\"stride_blocks\":%d,"
                        "\"competitors\":%d,\"pressure_seed\":%u,\"read_value\":%u,"
                        "\"probe_cycles\":%llu",
                        i, labels[int(c.pattern)], s.allocation, s.trial, block, s.anchor,
                        c.stride, c.competitors, s.seed, expected, samples[i].probe);
            if (matched) {
                const Sample& native = samples[schedule.size() + i];
                if (native.hot != expected) throw std::runtime_error("Paired read output mismatch");
                std::printf(",\"native_first\":%s,\"native_cycles\":%llu",
                            native_first(s) ? "true" : "false", native.probe);
            }
            std::puts("}");
            continue;
        }
        std::printf("{\"kind\":\"sample\",\"pattern\":\"%s\",\"allocation\":%d,\"trial\":%d,"
                    "\"allocation_base_block\":%llu,\"base_block\":%d,\"sector_word\":0,"
                    "\"stride_blocks\":%d,\"competitors\":%d,\"span_blocks\":%d,\"pressure_seed\":%u,"
                    "\"cold_cycles\":%llu,\"hot_cycles\":%llu,\"probe_cycles\":%llu}\n",
                    labels[int(c.pattern)], s.allocation, s.trial, block, s.anchor,
                    c.stride, c.competitors, c.stride * c.competitors, s.seed,
                    samples[i].cold, samples[i].hot, samples[i].probe);
    }
    return 0;
}

int run(const std::string& mode) {
    cudaDeviceProp device{};
    CUDA(cudaGetDeviceProperties(&device, 0));
    if (device.l2CacheSize < 128 * 128) throw std::runtime_error("Missing L2 geometry");
    const int capacity_blocks = device.l2CacheSize / (4 * block_words);
    if (!mode.empty()) return run_pressure(device, capacity_blocks, mode);
    std::vector<int> strides{1, 64, 512};
    for (int divisor : {128, 64, 32, 16, 8}) strides.push_back(capacity_blocks / divisor);
    std::sort(strides.begin(), strides.end());
    strides.erase(std::unique(strides.begin(), strides.end()), strides.end());
    std::vector<Condition> cases;
    for (int stride : strides)
        for (int n : {4, 8, 16, 24, 32, 48, 64}) cases.push_back({Pattern::strided, stride, n, 0});
    for (int offset : {0, 8, 16, 24}) cases.push_back({Pattern::neighbor, 0, 0, offset});
    cases.push_back({Pattern::capacity, 0, 0, 0});

    // Six anchors visited twice in each of three allocations, all held live.
    constexpr int anchors[] = {3, 74, 217, 465, 738, 1013};
    const size_t words = (size_t(1024) + size_t(64) * strides.back()) * block_words;
    const size_t bytes = words * sizeof(Word) + 4096;
    Buffer first(bytes), second(bytes), third(bytes);
    Buffer* buffers[buffer_count] = {&first, &second, &third};
    Word* tables[buffer_count]{};
    const int blocks = device.multiProcessorCount * 4;
    constexpr int threads = 256;
    for (int i = 0; i < buffer_count; ++i) {
        const auto aligned = (reinterpret_cast<std::uintptr_t>(buffers[i]->raw) + 4095) & ~std::uintptr_t(4095);
        tables[i] = reinterpret_cast<Word*>(aligned);
        initialize<<<blocks, threads>>>(tables[i], words, 257u * (i + 1));
    }
    const size_t scrub_bytes = size_t(device.l2CacheSize) * 2;
    Buffer scrub(scrub_bytes), sink(blocks * threads * sizeof(Word));
    initialize<<<blocks, threads>>>(scrub.as<Word>(), scrub_bytes / sizeof(Word), 2039);
    CUDA(cudaGetLastError());
    std::printf("{\"kind\":\"info\",\"device\":\"%s\",\"l2_bytes\":%d,\"scrub_bytes\":%zu,"
                "\"block_bytes\":128,\"virtual_witness_block_xor\":8,\"seed\":714203}\n",
                device.name, device.l2CacheSize, scrub_bytes);
    std::fflush(stdout);

    std::vector<Scheduled> schedule;
    for (int a = 0; a < buffer_count; ++a)
        for (int trial = 0; trial < trials; ++trial)
            for (Condition c : cases) schedule.push_back({a, trial, anchors[trial % 6], c});
    std::mt19937 rng(714203);
    std::shuffle(schedule.begin(), schedule.end(), rng);
    std::vector<Sample> samples(schedule.size());
    Buffer result(samples.size() * sizeof(Sample));
    auto flush_l2 = [&]() {
        scrub_cache<<<blocks, threads>>>(scrub.as<Word>(), scrub_bytes / sizeof(Word), sink.as<Word>());
    };
    for (size_t i = 0; i < schedule.size(); ++i) {
        const Scheduled& s = schedule[i];
        const Word* table = tables[s.allocation];
        Sample* output = result.as<Sample>() + i;
        flush_l2();
        measure<<<1, 1>>>(table, s.base_block, s.condition, output, false);
        if (s.condition.pattern == Pattern::capacity) {
            flush_l2();
            measure<<<1, 1>>>(table, s.base_block, s.condition, output, true);
        }
        CUDA(cudaGetLastError());
    }
    CUDA(cudaMemcpy(samples.data(), result.raw, samples.size() * sizeof(Sample), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < samples.size(); ++i) {
        const Scheduled& s = schedule[i];
        const Sample& value = samples[i];
        const char* name = s.condition.pattern == Pattern::strided ? "strided" :
                           s.condition.pattern == Pattern::neighbor ? "neighbor" : "capacity";
        const auto block = Cycles(reinterpret_cast<std::uintptr_t>(tables[s.allocation]) / 128);
        std::printf("{\"kind\":\"sample\",\"pattern\":\"%s\",\"allocation\":%d,\"trial\":%d,"
                    "\"allocation_base_block\":%llu,\"base_block\":%d,\"sector_word\":%d,"
                    "\"stride_blocks\":%d,\"competitors\":%d,\"cold_cycles\":%llu,"
                    "\"hot_cycles\":%llu,\"probe_cycles\":%llu}\n", name, s.allocation,
                    s.trial, block, s.base_block, s.condition.sector_word, s.condition.stride,
                    s.condition.competitors, value.cold, value.hot, value.probe);
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        const std::string mode = argc == 2 ? argv[1] : "";
        if (argc > 2 || (!mode.empty() && mode != "--pressure" && mode != "--counter-check"
                         && mode != "--matched-counter"))
            throw std::runtime_error("Usage: calibrate_l2 [--pressure | --counter-check | --matched-counter]");
        return run(mode);
    }
    catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
