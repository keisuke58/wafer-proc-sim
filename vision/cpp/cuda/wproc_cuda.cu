// wproc_cuda.cu — CUDA kernels for the wproc filters (optional backend).
//
// Mirrors the CPU implementations: clamped (replicated) borders, Gaussian
// radius = ceil(3*sigma), 3x3 Sobel with signed output. Compiled only when
// CMake finds a CUDA toolkit; see cuda_backend.hpp for the host-side API.
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "wproc/cuda_backend.hpp"

namespace wproc::cuda_backend {

namespace {

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                                 cudaGetErrorString(err));
}

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Horizontal Gaussian pass: each thread produces one output pixel.
__global__ void gauss_h(const float* in, float* out, int W, int H,
                        const float* k, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int i = -radius; i <= radius; ++i)
        acc += k[i + radius] * in[y * W + clampi(x + i, 0, W - 1)];
    out[y * W + x] = acc;
}

// Vertical Gaussian pass.
__global__ void gauss_v(const float* in, float* out, int W, int H,
                        const float* k, int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float acc = 0.0f;
    for (int i = -radius; i <= radius; ++i)
        acc += k[i + radius] * in[clampi(y + i, 0, H - 1) * W + x];
    out[y * W + x] = acc;
}

// 3x3 Sobel, both directions in one pass.
__global__ void sobel3(const float* in, float* gx, float* gy, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int xm = clampi(x - 1, 0, W - 1), xp = clampi(x + 1, 0, W - 1);
    int ym = clampi(y - 1, 0, H - 1), yp = clampi(y + 1, 0, H - 1);

    float tl = in[ym * W + xm], tc = in[ym * W + x], tr = in[ym * W + xp];
    float ml = in[y * W + xm], mr = in[y * W + xp];
    float bl = in[yp * W + xm], bc = in[yp * W + x], br = in[yp * W + xp];

    gx[y * W + x] = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
    gy[y * W + x] = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
}

// RAII device buffer so error paths can't leak.
struct DeviceBuf {
    float* ptr = nullptr;
    explicit DeviceBuf(std::size_t n) {
        check(cudaMalloc(&ptr, n * sizeof(float)), "cudaMalloc");
    }
    ~DeviceBuf() { cudaFree(ptr); }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
};

dim3 grid_for(int W, int H, dim3 block) {
    return dim3((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
}

}  // namespace

bool available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

Image gaussian_blur(const Image& src, float sigma) {
    if (sigma <= 0.0f)
        throw std::invalid_argument("cuda gaussian_blur: sigma must be > 0");
    if (!(sigma <= 1000.0f))
        throw std::invalid_argument("cuda gaussian_blur: sigma too large");
    const int W = src.width(), H = src.height();
    if (W == 0 || H == 0) return src;

    // Same kernel construction as the CPU path.
    const int radius = static_cast<int>(std::ceil(3.0f * sigma));
    std::vector<float> k(2 * radius + 1);
    float sum = 0.0f;
    const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int i = -radius; i <= radius; ++i) {
        float w = std::exp(-static_cast<float>(i) * i * inv2s2);
        k[i + radius] = w;
        sum += w;
    }
    for (float& w : k) w /= sum;

    const std::size_t n = static_cast<std::size_t>(W) * H;
    DeviceBuf d_in(n), d_tmp(n), d_out(n), d_k(k.size());
    check(cudaMemcpy(d_in.ptr, src.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice), "H2D image");
    check(cudaMemcpy(d_k.ptr, k.data(), k.size() * sizeof(float),
                     cudaMemcpyHostToDevice), "H2D kernel");

    dim3 block(16, 16);
    dim3 grid = grid_for(W, H, block);
    gauss_h<<<grid, block>>>(d_in.ptr, d_tmp.ptr, W, H, d_k.ptr, radius);
    check(cudaGetLastError(), "gauss_h launch");
    gauss_v<<<grid, block>>>(d_tmp.ptr, d_out.ptr, W, H, d_k.ptr, radius);
    check(cudaGetLastError(), "gauss_v launch");

    Image out(W, H);
    check(cudaMemcpy(out.data(), d_out.ptr, n * sizeof(float),
                     cudaMemcpyDeviceToHost), "D2H image");
    return out;
}

Gradient sobel(const Image& src) {
    const int W = src.width(), H = src.height();
    Gradient g{Image(W, H), Image(W, H)};
    if (W == 0 || H == 0) return g;

    const std::size_t n = static_cast<std::size_t>(W) * H;
    DeviceBuf d_in(n), d_gx(n), d_gy(n);
    check(cudaMemcpy(d_in.ptr, src.data(), n * sizeof(float),
                     cudaMemcpyHostToDevice), "H2D image");

    dim3 block(16, 16);
    sobel3<<<grid_for(W, H, block), block>>>(d_in.ptr, d_gx.ptr, d_gy.ptr, W, H);
    check(cudaGetLastError(), "sobel3 launch");

    check(cudaMemcpy(g.gx.data(), d_gx.ptr, n * sizeof(float),
                     cudaMemcpyDeviceToHost), "D2H gx");
    check(cudaMemcpy(g.gy.data(), d_gy.ptr, n * sizeof(float),
                     cudaMemcpyDeviceToHost), "D2H gy");
    return g;
}

}  // namespace wproc::cuda_backend
