#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int height = 610, width = 340, pixels = height * width, classes = 9;
constexpr int threads = 256, tile = 16, field_count = 8;
constexpr double mu = 1, tau = 0.02;

#define CUDA(op) do { cudaError_t status = (op); if (status != cudaSuccess) \
    throw std::runtime_error(std::string(#op) + ": " + cudaGetErrorString(status)); } while (0)

struct Fields {
    double* data[field_count]{};
};

extern "C" __global__ void q_kernel(Fields in, Fields out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= classes * pixels) return;
    double value = 0;
    for (int k = 0; k < field_count; k += 2) value += in.data[k][i] + in.data[k + 1][i];
    out.data[0][i] = 0.25 * value;
}

extern "C" __global__ void fixq_kernel(Fields in, const int* training, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int pixel = training[i];
    for (int k = 0; k < classes; ++k) in.data[0][k * pixels + pixel] = in.data[1][k * count + i];
}

extern "C" __global__ void nu_aux3_sum_kernel(Fields in, Fields out) {
    __shared__ double partial[threads];
    const int lane = threadIdx.x;
    const int column = blockIdx.x * blockDim.x + lane;
    partial[lane] = 0.0;
    if (column < pixels) {
        for (int k = 0; k < classes; ++k) {
            const int i = k * pixels + column;
            const double value = in.data[0][i] - in.data[1][i];
            out.data[0][i] = value;
            partial[lane] += value;
        }
    }
    __syncthreads();
    if (column < pixels) out.data[1][column] = partial[lane];
}

extern "C" __global__ void Vs_kernel(Fields in, Fields out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= classes * pixels) return;
    const double q = in.data[1][i];
    out.data[0][i] = (in.data[0][i] + mu * (q - in.data[2][i])) / (1 + mu);
    out.data[1][i] = fmax(q - in.data[3][i], 0.0);
    out.data[2][i] = in.data[6][i] + in.data[7][i % pixels];
    out.data[3][i] = q - in.data[5][i];
}

extern "C" __global__ void V4_kernel(Fields in, Fields out) {
    __shared__ double values[tile][tile + 1];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int x = blockIdx.x * tile + tx, y = blockIdx.y * tile + ty;
    if (x < height && y < width) {
        const int i = y * height + x;
        auto* input = reinterpret_cast<double2*>(in.data[0]);
        double2 v = input[i];
        v.x /= pixels;
        input[i].x = v.x;
        values[ty][tx] = v.x;
        reinterpret_cast<double2*>(out.data[0])[i] = v;
        reinterpret_cast<double2*>(out.data[1])[i] = v;
    }
    __syncthreads();
    const int row = blockIdx.x * tile + ty, col = blockIdx.y * tile + tx;
    if (row < height && col < width) out.data[2][row * width + col] = values[tx][ty];
}

extern "C" __global__ void V5_kernel(Fields in, Fields out, int pitch) {
    const int x = blockIdx.x * tile + threadIdx.x, y = blockIdx.y * tile + threadIdx.y;
    if (x >= height || y >= width) return;
    const int i = y * height + x, j = y * pitch + x;
    const double gh = reinterpret_cast<const double2*>(in.data[0])[i].x / pixels;
    const double gv = reinterpret_cast<const double2*>(in.data[1])[i].x / pixels;
    const double dh = in.data[2][j], dv = in.data[3][j];
    const double a = gh - dh, b = gv - dv;
    const double excess = fmax(sqrt(a*a + b*b) - tau, 0.0);
    const double scale = excess / (excess + tau);
    const double vh = scale * a, vv = scale * b;
    out.data[0][j] = vh;
    out.data[1][j] = vv;
    in.data[2][j] = dh - (gh - vh);
    in.data[3][j] = dv - (gv - vv);
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

template <class T> struct Buffer {
    T* ptr = nullptr;
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    ~Buffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t count) {
        CUDA(cudaMalloc(&ptr, count * sizeof(T)));
        if (reinterpret_cast<size_t>(ptr) % 32) throw std::runtime_error("Unaligned CUDA base");
    }
};

struct Shape {
    int rows, cols, pitch;
    size_t count() const { return size_t(rows) * pitch; }
};

struct Matrix {
    Buffer<double> gpu;
    Shape shape{};
    void setup(Shape s, std::istream* input = nullptr) {
        shape = s;
        gpu.allocate(s.count());
        if (!input) return;
        std::vector<double> host(s.count(), 0.0);
        for (int y = 0; y < s.rows; ++y)
            if (!input->read(reinterpret_cast<char*>(host.data() + size_t(y) * s.pitch), s.cols * sizeof(double)))
                throw std::runtime_error("Short input file");
        CUDA(cudaMemcpy(gpu.ptr, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    size_t validate(std::istream& reference, double& max_error) const {
        std::vector<double> actual(shape.count()), line(shape.cols);
        CUDA(cudaMemcpy(actual.data(), gpu.ptr, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
        for (int y = 0; y < shape.rows; ++y) {
            if (!reference.read(reinterpret_cast<char*>(line.data()), line.size() * sizeof(double)))
                throw std::runtime_error("Short CPU reference");
            for (int x = 0; x < shape.cols; ++x) {
                const double error = std::abs(actual[size_t(y) * shape.pitch + x] - line[x]);
                if (!std::isfinite(error) || error > 2e-10 * (1 + std::abs(line[x])))
                    throw std::runtime_error("CPU mismatch at row " + std::to_string(y) + ", column " + std::to_string(x));
                max_error = std::max(max_error, error);
            }
        }
        return size_t(shape.rows) * shape.cols;
    }
};

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("bench KERNEL PAD INPUT_DIR TRAINING_COUNT");
        const std::string kernel = argv[1], root = argv[3];
        const int pad = std::stoi(argv[2]), count = std::stoi(argv[4]);
        if (count < 1 || count > pixels || (pad != 0 && !(kernel == "V5_kernel" && pad == 2)))
            throw std::runtime_error("Invalid pad or training count");
        std::ifstream input(root + "/" + kernel + ".in", std::ios::binary);
        std::ifstream reference(root + "/" + kernel + ".ref", std::ios::binary);
        if (!input || !reference) throw std::runtime_error("Missing input or reference");
        const Shape plane{width, height, height + pad}, complex{width, 2 * height, 2 * height};
        const Shape transposed{height, width, width}, scores{classes, pixels, pixels}, bias{1, pixels, pixels};
        Matrix inputs[field_count], outputs[field_count];
        std::vector<Shape> in_shapes, out_shapes;
        Buffer<int> labels;
        if (kernel == "q_kernel") {
            in_shapes.assign(8, scores); out_shapes = {scores};
        } else if (kernel == "fixq_kernel") {
            in_shapes = {scores, {classes, count, count}};
            std::ifstream index(root + "/training_indices.i32", std::ios::binary);
            std::vector<int> host(count);
            if (!index.read(reinterpret_cast<char*>(host.data()), count * sizeof(int)) ||
                !std::is_sorted(host.begin(), host.end()) ||
                std::adjacent_find(host.begin(), host.end()) != host.end() ||
                host.front() < 0 || host.back() >= pixels) throw std::runtime_error("Invalid training indices");
            labels.allocate(count);
            CUDA(cudaMemcpy(labels.ptr, host.data(), count * sizeof(int), cudaMemcpyHostToDevice));
        } else if (kernel == "nu_aux3_sum_kernel") {
            in_shapes = {scores, scores}; out_shapes = {scores, bias};
        } else if (kernel == "Vs_kernel") {
            in_shapes = {scores, scores, scores, scores, scores, scores, scores, bias};
            out_shapes = {scores, scores, scores, scores};
        } else if (kernel == "V4_kernel") {
            in_shapes = {complex}; out_shapes = {complex, complex, transposed};
        } else if (kernel == "V5_kernel") {
            in_shapes = {complex, complex, plane, plane}; out_shapes = {plane, plane};
        } else throw std::runtime_error("Unknown kernel");
        cudaDeviceProp gpu{};
        CUDA(cudaGetDeviceProperties(&gpu, 0));
        constexpr int scrub_blocks = 256;
        const size_t scrub_bytes = size_t(gpu.l2CacheSize) * 4;
        size_t allocated = scrub_bytes + scrub_blocks * threads * sizeof(unsigned) +
                           (labels.ptr ? count * sizeof(int) : 0);
        for (const auto& shape : in_shapes) allocated += shape.count() * sizeof(double);
        for (const auto& shape : out_shapes) allocated += shape.count() * sizeof(double);
        if (allocated > 512ull * 1024 * 1024) throw std::runtime_error("Allocation budget exceeded");
        Fields in, out;
        for (size_t i = 0; i < in_shapes.size(); ++i) {
            inputs[i].setup(in_shapes[i], &input);
            in.data[i] = inputs[i].gpu.ptr;
        }
        if (input.peek() != EOF) throw std::runtime_error("Extra input data");
        for (size_t i = 0; i < out_shapes.size(); ++i) {
            outputs[i].setup(out_shapes[i]);
            out.data[i] = outputs[i].gpu.ptr;
            CUDA(cudaMemset(out.data[i], 0xff, out_shapes[i].count() * sizeof(double)));
        }
        Buffer<unsigned> stream, sink;
        stream.allocate(scrub_bytes / sizeof(unsigned));
        sink.allocate(scrub_blocks * threads);
        CUDA(cudaMemset(stream.ptr, 0xa7, scrub_bytes));
        cudaEvent_t begin, end;
        CUDA(cudaEventCreate(&begin)); CUDA(cudaEventCreate(&end));
        condition<<<scrub_blocks, threads>>>(stream.ptr, scrub_bytes / sizeof(unsigned), sink.ptr);
        CUDA(cudaGetLastError());
        const dim3 block(tile, tile), grid((height + tile - 1) / tile, (width + tile - 1) / tile);
        CUDA(cudaEventRecord(begin));
        if (kernel == "q_kernel") q_kernel<<<(classes * pixels + threads - 1) / threads, threads>>>(in, out);
        else if (kernel == "fixq_kernel") fixq_kernel<<<(count + threads - 1) / threads, threads>>>(in, labels.ptr, count);
        else if (kernel == "nu_aux3_sum_kernel") nu_aux3_sum_kernel<<<(pixels + threads - 1) / threads, threads>>>(in, out);
        else if (kernel == "Vs_kernel") Vs_kernel<<<(classes * pixels + threads - 1) / threads, threads>>>(in, out);
        else if (kernel == "V4_kernel") V4_kernel<<<grid, block>>>(in, out);
        else V5_kernel<<<grid, block>>>(in, out, plane.pitch);
        CUDA(cudaGetLastError());
        CUDA(cudaEventRecord(end)); CUDA(cudaEventSynchronize(end));
        float ms = 0;
        CUDA(cudaEventElapsedTime(&ms, begin, end));
        CUDA(cudaEventDestroy(begin)); CUDA(cudaEventDestroy(end));
        double max_error = 0;
        size_t cells = 0;
        if (kernel == "fixq_kernel" || kernel == "V4_kernel") cells += inputs[0].validate(reference, max_error);
        for (size_t i = 0; i < out_shapes.size(); ++i) cells += outputs[i].validate(reference, max_error);
        if (kernel == "V5_kernel") {
            cells += inputs[2].validate(reference, max_error);
            cells += inputs[3].validate(reference, max_error);
        }
        if (reference.peek() != EOF) throw std::runtime_error("Extra CPU reference data");
        std::printf("{\"kernel\":\"%s\",\"pad\":%d,\"ms\":%.9g,\"cells\":%zu,\"max_error\":%.9g,\"peak_bytes\":%zu}\n",
                    argv[1], pad, ms, cells, max_error, allocated);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

