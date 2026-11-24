// Uniform row-tail padding: select the pitch before launching either layout.
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Word = std::uint32_t;

constexpr int window_words = 8;
constexpr int original_second_window = 24;
constexpr int assumed_tag_bytes = 128;
constexpr int tag_words = assumed_tag_bytes / sizeof(Word);
constexpr int rounds = 40;
constexpr int threads = 256;
constexpr size_t mib = 1024 * 1024;
constexpr size_t extra_budget = 8 * mib;
// Explicit host allocation policy: pass these rounded byte counts to cudaMalloc.
constexpr size_t layout_granule = 2 * mib;

void check(cudaError_t code, const char* expression) {
    if (code != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(code));
    }
}
#define CUDA(call) check((call), #call)

__host__ __device__ Word field_value(int row, int column) {
    return (2654435761u * (Word(row) + 1u)) ^ (2246822519u * (Word(column) + 1u));
}

__global__ void init_layout(Word* table, int rows, int pitch, int columns) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < size_t(rows) * pitch; i += gridDim.x * blockDim.x) {
        int col = int(i % pitch);
        table[i] = col < columns ? field_value(int(i / pitch), col) : 0xdeadc0deu;
    }
}

__device__ __forceinline__ Word load_cg(const Word* address) {
    Word result;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(result) : "l"(address) : "memory");
    return result;
}

__global__ void condition_cache(const Word* scrub, int count, Word* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Word sum = 0;
    for (int i = tid; i < count; i += gridDim.x * blockDim.x) sum += load_cg(scrub + i);
    output[tid] = sum;
}

__device__ __forceinline__ Word scan_once(const Word* table, int rows, int pitch, int second_window) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid % window_words;
    int group = tid / window_words;
    int groups = gridDim.x * blockDim.x / window_words;
    Word sum = 0;
    for (int row = group; row < rows; row += groups) {
        const Word* first = table + size_t(row) * pitch + lane;
        Word a = load_cg(first);
        Word b = load_cg(first + second_window);
        sum += a + b;
    }
    return sum;
}

extern "C" __global__ void row_tail_warm(const Word* table, int rows, int pitch, int second_window, Word* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = scan_once(table, rows, pitch, second_window);
}

extern "C" __global__ void row_tail_target(const Word* table, int rows, int pitch, int second_window, Word* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Word sum = 0;
    for (int pass = 0; pass < rounds; ++pass) {
        sum += scan_once(table, rows, pitch, second_window);
    }
    output[tid] = sum;
}

struct Plan {
    cudaDeviceProp device{};
    std::string case_name;
    int columns = 56;
    int second_window_words = original_second_window;
    size_t extra_budget_bytes = extra_budget;
    int rows = 0;
    int blocks = 0;
    size_t scrub_bytes = 0;
    int explicit_pad = -1;
    int pad() const { return explicit_pad >= 0 ? explicit_pad : (tag_words - columns % tag_words) % tag_words; }
    size_t tag_count(int pitch) const {
        // Each 8-word window is sector-aligned and fits in one 128-byte block.
        // Rows share no blocks across these two windows at the supported strides.
        size_t blocks_touched = 0;
        for (int row = 0; row < rows; ++row) {
            const size_t first = size_t(row) * pitch;
            const size_t second = first + second_window_words;
            blocks_touched += 1 + (first / tag_words != second / tag_words);
        }
        return blocks_touched;
    }
    size_t output_bytes() const { return size_t(blocks) * threads * sizeof(Word); }
    size_t allocation(int pitch) const {
        const size_t payload = size_t(rows) * pitch * sizeof(Word);
        return ((payload + layout_granule - 1) / layout_granule) * layout_granule;
    }
};

int read_case_number(const char*& begin, const char* end, bool last) {
    if (begin == end || *begin < '1' || *begin > '9')
        throw std::invalid_argument("Expected a positive decimal integer in family-W-G-F");
    int value;
    auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || (last ? parsed.ptr != end : parsed.ptr == end || *parsed.ptr != '-'))
        throw std::invalid_argument("Expected family-W-G-F");
    begin = last ? end : parsed.ptr + 1;
    return value;
}

Plan make_plan(const std::string& case_name, int explicit_pad) {
    Plan p;
    p.case_name = case_name;
    p.explicit_pad = explicit_pad;
    CUDA(cudaSetDevice(0));
    CUDA(cudaGetDeviceProperties(&p.device, 0));
    int l2 = p.device.l2CacheSize;
    if (l2 <= 0 || p.device.warpSize != 32) {
        throw std::runtime_error("Experiment requires a reported L2 capacity and 32-lane warps");
    }
    // Each case specifies its row count relative to the reported L2 capacity.
    size_t footprint = 0;
    if (case_name == "original") footprint = size_t(l2) * 3 / 4;
    else if (case_name == "holdout") footprint = size_t(l2) * 4 / 5;
    else if (case_name == "w72_mid") {
        p.columns = 72;
        p.extra_budget_bytes = size_t(l2);
        footprint = size_t(l2) * 11 / 16;
    } else if (case_name == "w80_high") {
        p.columns = 80;
        p.extra_budget_bytes = size_t(l2);
        footprint = size_t(l2) * 13 / 16;
    } else if (case_name.compare(0, 7, "family-") == 0) {
        const char* begin = case_name.data() + 7;
        const char* end = case_name.data() + case_name.size();
        p.columns = read_case_number(begin, end, false);
        p.second_window_words = read_case_number(begin, end, false);
        const int percent = read_case_number(begin, end, true);
        if (p.columns < 64 || p.columns > 4096 || p.columns % window_words || !(p.columns % tag_words)
                || (p.second_window_words != 8 && p.second_window_words != 16 && p.second_window_words != 24)
                || percent > 100) {
            throw std::invalid_argument("W must be 64..4096 and divisible by 8 but not 32; "
                                        "G must be 8, 16 or 24; F must be 1..100");
        }
        p.extra_budget_bytes = size_t(l2) + layout_granule;
        footprint = size_t(l2) * percent / 100;
    } else throw std::invalid_argument("Unknown target case: " + case_name);
    p.rows = int(footprint / assumed_tag_bytes / threads * threads);
    p.scrub_bytes = size_t(l2) * 2;
    if (!p.rows || size_t(p.rows) * (p.columns + p.pad()) > std::numeric_limits<int>::max()
        || p.scrub_bytes / sizeof(Word) > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Device dimensions outside supported range");
    }
    int active_blocks = 0;
    CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, row_tail_target, threads, 0));
    if (active_blocks < 1) throw std::runtime_error("No resident blocks available");
    p.blocks = p.device.multiProcessorCount * std::min(active_blocks, 2);
    if (p.allocation(p.columns + p.pad()) - p.allocation(p.columns) > p.extra_budget_bytes) {
        throw std::runtime_error("Derived padding exceeds the registered extra allocation budget");
    }
    return p;
}

void print_plan(const Plan& p) {
    std::printf("{\"kind\":\"plan\",\"case\":\"%s\",\"device\":\"%s\",\"cc_major\":%d,\"cc_minor\":%d,"
                "\"l2_bytes\":%d,\"rows\":%d,\"rounds\":%d,\"logical_words\":%d,\"pad_words\":%d,"
                "\"second_window_words\":%d,\"assumed_tag_bytes\":%d,\"warmup_rounds\":1,\"blocks\":%d,\"threads\":%d,"
                "\"grid_barrier\":false,\"layout_granule_bytes\":%zu,\"extra_budget_bytes\":%zu,"
                "\"base_allocation_bytes\":%zu,\"pad_allocation_bytes\":%zu,"
                "\"scrub_bytes\":%zu,\"output_bytes\":%zu,\"base_tag_count\":%zu,"
                "\"pad_tag_count\":%zu,\"source_sectors_per_sweep\":%zu}\n",
                p.case_name.c_str(), p.device.name, p.device.major, p.device.minor, p.device.l2CacheSize, p.rows,
                rounds, p.columns, p.pad(), p.second_window_words, assumed_tag_bytes, p.blocks, threads,
                layout_granule, p.extra_budget_bytes,
                p.allocation(p.columns), p.allocation(p.columns + p.pad()), p.scrub_bytes,
                p.output_bytes(), p.tag_count(p.columns), p.tag_count(p.columns + p.pad()), size_t(p.rows) * 2);
    std::fflush(stdout);
}

struct Buffer {
    Word* data = nullptr;
    explicit Buffer(size_t bytes) { CUDA(cudaMalloc(&data, bytes)); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Event {
    cudaEvent_t id{};
    Event() { CUDA(cudaEventCreate(&id)); }
    ~Event() { cudaEventDestroy(id); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

std::vector<Word> reference(const Plan& plan) {
    const int worker_count = plan.blocks * threads;
    std::vector<Word> sums(worker_count, 0);
    for (int row = 0; row < plan.rows; ++row) {
        int leader = (row % (worker_count / window_words)) * window_words;
        for (int lane = 0; lane < window_words; ++lane) {
            Word pair = field_value(row, lane) + field_value(row, plan.second_window_words + lane);
            sums[leader + lane] += rounds * pair;
        }
    }
    return sums;
}

size_t free_bytes() {
    size_t available, total;
    CUDA(cudaMemGetInfo(&available, &total));
    return available;
}

void prepare_layout(const Plan& p, int pitch, const Buffer& table,
                    const Buffer& scrub, const Buffer& output) {
    if (reinterpret_cast<std::uintptr_t>(table.data) % assumed_tag_bytes != 0) {
        throw std::runtime_error("Table base is not aligned to assumed tag");
    }
    init_layout<<<p.blocks, threads>>>(table.data, p.rows, pitch, p.columns);
    CUDA(cudaGetLastError());
    CUDA(cudaMemset(scrub.data, 0x5a, p.scrub_bytes));
    condition_cache<<<p.blocks, threads>>>(scrub.data, int(p.scrub_bytes / sizeof(Word)), output.data);
    CUDA(cudaGetLastError());
    row_tail_warm<<<p.blocks, threads>>>(table.data, p.rows, pitch, p.second_window_words, output.data);
    CUDA(cudaGetLastError());
    CUDA(cudaDeviceSynchronize());
}

float finish_timing(const Event& start, const Event& stop) {
    CUDA(cudaEventRecord(stop.id));
    CUDA(cudaEventSynchronize(stop.id));
    float ms = 0;
    CUDA(cudaEventElapsedTime(&ms, start.id, stop.id));
    if (!(ms > 0)) throw std::runtime_error("Invalid CUDA event duration");
    return ms;
}

void validate_output(const Plan& p, const Buffer& output, const std::vector<Word>& expected) {
    std::vector<Word> actual(expected.size());
    CUDA(cudaMemcpy(actual.data(), output.data, p.output_bytes(), cudaMemcpyDeviceToHost));
    if (actual != expected) throw std::runtime_error("Kernel output differs from logical CPU oracle");
}

void run_arm(const Plan& p, const std::vector<Word>& expected, int pad_words) {
    int pitch = p.columns + pad_words;
    const size_t allocation = p.allocation(pitch);
    size_t before = free_bytes();
    Buffer table(allocation);
    size_t after_table = free_bytes();
    Buffer scrub(p.scrub_bytes);
    Buffer output(p.output_bytes());
    Event start, stop;
    prepare_layout(p, pitch, table, scrub, output);
    size_t all_live = free_bytes();

    // Exactly one timed target. Each CTA rereads its own rows in the same order.
    CUDA(cudaEventRecord(start.id));
    row_tail_target<<<p.blocks, threads>>>(table.data, p.rows, pitch, p.second_window_words, output.data);
    CUDA(cudaGetLastError());
    float ms = finish_timing(start, stop);
    validate_output(p, output, expected);
    if (p.columns != 56) {
        // Verify the entire live input after timing, including untouched padding.
        constexpr int batch_rows = 1024;
        std::vector<Word> batch(size_t(batch_rows) * pitch);
        for (int first = 0; first < p.rows; first += batch_rows) {
            int count = std::min(batch_rows, p.rows - first);
            CUDA(cudaMemcpy(batch.data(), table.data + size_t(first) * pitch,
                            size_t(count) * pitch * sizeof(Word), cudaMemcpyDeviceToHost));
            for (int r = 0; r < count; ++r)
                for (int col = 0; col < pitch; ++col) {
                    Word value = col < p.columns ? field_value(first + r, col) : 0xdeadc0deu;
                    if (batch[size_t(r) * pitch + col] != value)
                        throw std::runtime_error("Logical field or padding sentinel differs");
                }
        }
    }
    std::printf("{\"kind\":\"measurement\",\"arm\":\"%s\",\"rows\":%d,\"rounds\":%d,"
                "\"pitch_words\":%d,\"pad_words\":%d,\"kernel_ms\":%.9g,\"verified\":true,"
                "\"checked_outputs\":%zu,\"allocation_bytes\":%zu,\"peak_bytes\":%zu,"
                "\"device_free_before\":%zu,\"device_free_after_table\":%zu,"
                "\"device_free_all_live\":%zu}\n",
                pad_words ? "pad" : "base", p.rows, rounds, pitch, pad_words, ms,
                expected.size(), allocation, allocation + p.scrub_bytes + p.output_bytes(),
                before, after_table, all_live);
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    try {
        std::string mode = argc == 1 ? "both" : argv[1];
        std::string case_name = argc > 2 ? argv[2] : "original";
        if (argc > 4 || (mode != "both" && mode != "base" && mode != "pad"
                         && mode != "info" && mode != "--info")) {
            throw std::runtime_error("Usage: bench [both|base|pad|info] [case] [explicit pad words]");
        }
        const int explicit_pad = argc == 4 ? std::stoi(argv[3]) : -1;
        if (argc == 4 && (explicit_pad < 0 || explicit_pad >= tag_words || explicit_pad % window_words))
            throw std::invalid_argument("Explicit padding must be one of 0, 8, 16, 24 words");
        const Plan plan = make_plan(case_name, explicit_pad);
        print_plan(plan);
        if (mode == "info" || mode == "--info") return 0;
        auto expected = reference(plan);
        if (mode == "both" || mode == "base") run_arm(plan, expected, 0);
        if (mode == "both" || mode == "pad") run_arm(plan, expected, plan.pad());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
