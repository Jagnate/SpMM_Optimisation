// cublas_gemm_bin.cu
// Read raw float32 binary matrices A_dense.bin (row-major) and B.bin (row-major),
// run C = A * B on GPU using cuBLAS, time the GPU kernel, and write C_out.bin (row-major).
// Supports FP32 path or FP16 inputs with FP32 accumulation (Tensor Core path).
//
// Build:
//   nvcc -O2 cublas_gemm_bin.cu -lcublas -o cublas_gemm_bin
//
// Run example:
//   ./cublas_gemm_bin --a case_out/A_dense.bin --b case_out/B.bin --m 1024 --k 1024 --n 1024 --fp16 --repeat 5
//
// Output:
//   prints average GPU time (s) and writes C_out.bin (raw float32 row-major).
//
// Notes:
//  - Input raw files must be float32 row-major binary (no header), shape must match --m,--k,--n.
//  - If --fp16 is set, inputs are cast to fp16 on host before upload and cublasGemmEx uses
//    CUBLAS_COMPUTE_32F_FAST_16F with CUBLAS_GEMM_DEFAULT_TENSOR_OP to favor Tensor Cores.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

static bool read_raw_f32(const char* path, std::vector<float>& out, size_t elems) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::perror(path);
        return false;
    }
    out.resize(elems);
    size_t r = std::fread(out.data(), sizeof(float), elems, f);
    std::fclose(f);
    if (r != elems) {
        std::fprintf(stderr, "read_raw_f32: expected %zu elements but read %zu from %s\n", elems, r, path);
        return false;
    }
    return true;
}

static bool write_raw_f32(const char* path, const std::vector<float>& data) {
    FILE* f = std::fopen(path, "wb");
    if (!f) {
        std::perror(path);
        return false;
    }
    size_t w = std::fwrite(data.data(), sizeof(float), data.size(), f);
    std::fclose(f);
    if (w != data.size()) {
        std::fprintf(stderr, "write_raw_f32: wrote %zu of %zu elements\n", w, data.size());
        return false;
    }
    return true;
}

// convert row-major -> column-major (float)
static void row_to_col_f32(const std::vector<float>& r, std::vector<float>& c, int rows, int cols) {
    c.resize((size_t)rows * cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            c[(size_t)j * rows + i] = r[(size_t)i * cols + j];
}

int main(int argc, char** argv) {
    std::string a_file, b_file, out_file = "C_out.bin";
    int M=0, K=0, N=0;
    bool use_fp16=false;
    int repeat = 3;

    for (int i=1;i<argc;++i) {
        std::string s = argv[i];
        if (s=="--a") { a_file = argv[++i]; continue; }
        if (s=="--b") { b_file = argv[++i]; continue; }
        if (s=="--out") { out_file = argv[++i]; continue; }
        if (s=="--m") { M = atoi(argv[++i]); continue; }
        if (s=="--k") { K = atoi(argv[++i]); continue; }
        if (s=="--n") { N = atoi(argv[++i]); continue; }
        if (s=="--fp16") { use_fp16 = true; continue; }
        if (s=="--repeat") { repeat = atoi(argv[++i]); continue; }
    }

    if (a_file.empty() || b_file.empty() || M<=0 || K<=0 || N<=0) {
        std::fprintf(stderr, "Usage: %s --a A_dense.bin --b B.bin --m M --k K --n N [--out C_out.bin] [--fp16] [--repeat R]\n", argv[0]);
        return 1;
    }

    size_t elemsA = (size_t)M * K;
    size_t elemsB = (size_t)K * N;

    std::vector<float> A_row;
    std::vector<float> B_row;
    if (!read_raw_f32(a_file.c_str(), A_row, elemsA)) return 2;
    if (!read_raw_f32(b_file.c_str(), B_row, elemsB)) return 2;

    // Convert to column-major for cuBLAS
    std::vector<float> A_col, B_col;
    row_to_col_f32(A_row, A_col, M, K);
    row_to_col_f32(B_row, B_col, K, N);

    // Setup CUDA / cuBLAS
    cudaSetDevice(0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    void *dA=nullptr, *dB=nullptr, *dC=nullptr;
    size_t sizeC = (size_t)M * N * sizeof(float);
    cudaMalloc(&dC, sizeC);
    cudaMemset(dC, 0, sizeC);

    if (use_fp16) {
        // convert host column-major float -> __half arrays
        std::vector<__half> Ah(elemsA), Bh(elemsB);
        for (size_t i=0;i<elemsA;++i) Ah[i] = __float2half(A_col[i]);
        for (size_t i=0;i<elemsB;++i) Bh[i] = __float2half(B_col[i]);
        cudaMalloc(&dA, elemsA * sizeof(__half));
        cudaMalloc(&dB, elemsB * sizeof(__half));
        cudaMemcpy(dA, Ah.data(), elemsA * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, Bh.data(), elemsB * sizeof(__half), cudaMemcpyHostToDevice);
    } else {
        cudaMalloc(&dA, elemsA * sizeof(float));
        cudaMalloc(&dB, elemsB * sizeof(float));
        cudaMemcpy(dA, A_col.data(), elemsA * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B_col.data(), elemsB * sizeof(float), cudaMemcpyHostToDevice);
    }

    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;

    // Warmup
    if (use_fp16) {
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha_f,
            dA, CUDA_R_16F, M,
            dB, CUDA_R_16F, K,
            &beta_f,
            dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    } else {
        cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha_f,
            (const float*)dA, M,
            (const float*)dB, K,
            &beta_f,
            (float*)dC, M);
    }
    cudaDeviceSynchronize();

    double avg_gpu_s = 0.0;
    for (int it=0; it<repeat; ++it) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        if (use_fp16) {
            cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha_f,
                dA, CUDA_R_16F, M,
                dB, CUDA_R_16F, K,
                &beta_f,
                dC, CUDA_R_32F, M,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        } else {
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha_f,
                (const float*)dA, M,
                (const float*)dB, K,
                &beta_f,
                (float*)dC, M);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        avg_gpu_s += (double)ms * 1e-3;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    avg_gpu_s /= (double)repeat;

    // retrieve C (column-major) and convert to row-major
    std::vector<float> C_col((size_t)M * N);
    cudaMemcpy(C_col.data(), dC, sizeC, cudaMemcpyDeviceToHost);
    std::vector<float> C_row((size_t)M * N);
    for (int i=0;i<M;++i)
        for (int j=0;j<N;++j)
            C_row[(size_t)i * N + j] = C_col[(size_t)j * M + i];

    // write C_out.bin
    if (!write_raw_f32(out_file.c_str(), C_row)) {
        std::fprintf(stderr, "Failed to write %s\n", out_file.c_str());
    }

    std::cout << "Average GPU time (s): " << avg_gpu_s << "\n";
    std::cout << "Wrote " << out_file << " (float32 row-major)\n";

    // cleanup
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    cublasDestroy(handle);
    return 0;
}