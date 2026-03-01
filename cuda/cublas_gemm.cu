// cublas_gemm.cu
// Simple cuBLAS dense GEMM runner that reads A.npy and B.npy (float32, row-major),
// runs C = A * B on GPU (FP32 or FP16 path), times the GPU kernel, and writes C_out.bin.
//
// Build:
//   nvcc -O2 cublas_gemm.cu -lcublas -o cublas_gemm
//
// Run:
//   ./cublas_gemm --a case_out/A_dense.npy --b case_out/B.npy --m 1024 --k 1024 --n 1024 --fp16 --repeat 5
//
// Output:
//   - prints GPU time (average over repeats)
//   - writes C_out.bin (raw float32 row-major) which you can load in Python:
//       C = np.fromfile("C_out.bin", dtype=np.float32).reshape((m,n))
//
// Notes:
//  - This loader implements a minimal .npy parser for common numpy versions and
//    supports float32 and float16 arrays saved in row-major (C-order).
//  - We convert row-major -> column-major before calling cuBLAS (cuBLAS expects column-major).
//  - When using --fp16, inputs are cast to FP16 and cuBLAS compute uses CUBLAS_COMPUTE_32F_FAST_16F
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Minimal NPY loader for float32/float16 (C-order). Not fully general,
// but handles files produced by numpy.save(..., allow_pickle=False).
// Returns true on success and fills shape and data (as float32 vector).
bool load_npy_float(const std::string &filename, std::vector<int> &shape, std::vector<float> &out_f32) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        std::cerr << "Failed to open " << filename << "\n";
        return false;
    }
    // Read magic string
    char magic[6];
    ifs.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::cerr << "Not a .npy file: " << filename << "\n";
        return false;
    }
    uint8_t major = ifs.get();
    uint8_t minor = ifs.get();
    uint16_t header_len16 = 0;
    uint32_t header_len32 = 0;
    size_t header_len = 0;
    if (major == 1) {
        ifs.read(reinterpret_cast<char*>(&header_len16), 2);
        header_len = header_len16;
    } else {
        // version 2/3
        ifs.read(reinterpret_cast<char*>(&header_len32), 4);
        header_len = header_len32;
    }
    std::string header;
    header.resize(header_len);
    ifs.read(&header[0], header_len);

    // Parse header: look for descr, fortran_order, shape
    auto find_token = [&](const std::string &key)->std::string {
        auto p = header.find(key);
        if (p == std::string::npos) return "";
        auto q = header.find(':', p);
        if (q == std::string::npos) return "";
        auto end = header.find(',', q);
        if (end == std::string::npos) end = header.find('}', q);
        if (end == std::string::npos) return "";
        return header.substr(q+1, end-q-1);
    };

    // crude parse for descr
    std::string descr;
    {
        auto p = header.find("'descr'");
        if (p == std::string::npos) p = header.find("\"descr\"");
        if (p != std::string::npos) {
            auto colon = header.find(':', p);
            auto quote = header.find('\'', colon+1);
            if (quote == std::string::npos) quote = header.find('\"', colon+1);
            if (quote != std::string::npos) {
                auto quote2 = header.find(header[quote], quote+1);
                if (quote2 != std::string::npos) {
                    descr = header.substr(quote+1, quote2-quote-1);
                }
            }
        }
    }
    bool fortran_order = false;
    {
        auto p = header.find("'fortran_order'");
        if (p == std::string::npos) p = header.find("\"fortran_order\"");
        if (p != std::string::npos) {
            auto colon = header.find(':', p);
            auto valpos = header.find_first_not_of(" \t", colon+1);
            if (valpos != std::string::npos) {
                if (header[valpos] == 'T' || header[valpos] == 't') fortran_order = true;
            }
        }
    }
    // parse shape tuple (e.g. (1024, 1024))
    {
        auto p = header.find('(');
        auto q = header.find(')');
        if (p != std::string::npos && q != std::string::npos && q > p) {
            std::string inside = header.substr(p+1, q-p-1);
            shape.clear();
            std::istringstream ss(inside);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                // trim
                auto a = tok.find_first_not_of(" \t");
                if (a == std::string::npos) continue;
                auto b = tok.find_last_not_of(" \t");
                std::string t = tok.substr(a, b-a+1);
                if (t.size()==0) continue;
                try {
                    int v = std::stoi(t);
                    shape.push_back(v);
                } catch (...) { }
            }
            if (shape.size()==0) {
                // scalar?
                shape.push_back(1);
            }
        } else {
            std::cerr << "Failed to parse shape in header: " << header << "\n";
            return false;
        }
    }

    if (fortran_order) {
        std::cerr << "Fortran-order .npy not supported by this loader.\n";
        return false;
    }
    if (descr.size() < 3) {
        std::cerr << "Failed to parse descr from header\n";
        return false;
    }
    // descr example: '<f8' or '<f4' or '<f2'
    char endian = descr[0];
    char typec = descr[1];
    int bytes = stoi(descr.substr(2));
    size_t nelem = 1;
    for (int d : shape) nelem *= d;

    if (!(typec == 'f' && (bytes == 4 || bytes == 2))) {
        std::cerr << "Only float32/float16 .npy supported. descr=" << descr << "\n";
        return false;
    }

    // Now read raw data
    out_f32.resize(nelem);
    if (bytes == 4) {
        // float32
        ifs.read(reinterpret_cast<char*>(out_f32.data()), nelem * 4);
        if (!ifs) {
            std::cerr << "Failed to read float32 data\n";
            return false;
        }
    } else {
        // float16: read into buffer of uint16, convert to float32
        std::vector<uint16_t> buf(nelem);
        ifs.read(reinterpret_cast<char*>(buf.data()), nelem * 2);
        if (!ifs) {
            std::cerr << "Failed to read float16 data\n";
            return false;
        }
        // convert half->float
        for (size_t i = 0; i < nelem; ++i) {
            __half h;
            uint16_t bits = buf[i];
            std::memcpy(&h, &bits, 2);
            out_f32[i] = __half2float(h);
        }
    }
    return true;
}

// convert row-major (r[i * cols + j]) to column-major (c[j * rows + i])
template<typename T>
void row_to_col(const T *r, T *c, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            c[(size_t)j * rows + i] = r[(size_t)i * cols + j];
}

int main(int argc, char **argv) {
    std::string a_file = "";
    std::string b_file = "";
    int M = 0, K = 0, N = 0;
    bool use_fp16 = false;
    int repeat = 3;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--a") { a_file = argv[++i]; continue; }
        if (s == "--b") { b_file = argv[++i]; continue; }
        if (s == "--m") { M = atoi(argv[++i]); continue; }
        if (s == "--k") { K = atoi(argv[++i]); continue; }
        if (s == "--n") { N = atoi(argv[++i]); continue; }
        if (s == "--fp16") { use_fp16 = true; continue; }
        if (s == "--repeat") { repeat = atoi(argv[++i]); continue; }
    }
    if (a_file.empty() || b_file.empty() || M==0 || K==0 || N==0) {
        std::cerr << "Usage: " << argv[0] << " --a A.npy --b B.npy --m M --k K --n N [--fp16] [--repeat R]\n";
        return 1;
    }

    std::vector<int> ashape, bshape;
    std::vector<float> Arow, Brow;
    if (!load_npy_float(a_file, ashape, Arow)) return 1;
    if (!load_npy_float(b_file, bshape, Brow)) return 1;
    if ((int)ashape.size() != 2 || (int)bshape.size() != 2) {
        std::cerr << "Only 2D arrays supported\n"; return 1;
    }
    if (ashape[0] != M || ashape[1] != K || bshape[0] != K || bshape[1] != N) {
        std::cerr << "Shape mismatch vs M,K,N\n";
        std::cerr << "A shape from file: ("<<ashape[0]<<","<<ashape[1]<<")  expected ("<<M<<","<<K<<")\n";
        std::cerr << "B shape from file: ("<<bshape[0]<<","<<bshape[1]<<")  expected ("<<K<<","<<N<<")\n";
        return 1;
    }

    // Convert row-major -> column-major for cuBLAS
    std::vector<float> A_col((size_t)M * K);
    std::vector<float> B_col((size_t)K * N);
    row_to_col<float>(Arow.data(), A_col.data(), M, K);
    row_to_col<float>(Brow.data(), B_col.data(), K, N);

    // Allocate device arrays
    cudaSetDevice(0);
    cublasHandle_t cublas;
    cublasCreate(&cublas);

    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t sizeA, sizeB, sizeC;
    if (use_fp16) {
        // convert to half on host
        std::vector<__half> A_h((size_t)M * K), B_h((size_t)K * N);
        for (size_t i = 0; i < A_h.size(); ++i) A_h[i] = __float2half(A_col[i]);
        for (size_t i = 0; i < B_h.size(); ++i) B_h[i] = __float2half(B_col[i]);
        sizeA = A_h.size() * sizeof(__half);
        sizeB = B_h.size() * sizeof(__half);
        sizeC = (size_t)M * N * sizeof(float); // output as FP32 accumulate
        cudaMalloc(&dA, sizeA); cudaMalloc(&dB, sizeB); cudaMalloc(&dC, sizeC);
        cudaMemcpy(dA, A_h.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B_h.data(), sizeB, cudaMemcpyHostToDevice);
    } else {
        sizeA = (size_t)M * K * sizeof(float);
        sizeB = (size_t)K * N * sizeof(float);
        sizeC = (size_t)M * N * sizeof(float);
        cudaMalloc(&dA, sizeA); cudaMalloc(&dB, sizeB); cudaMalloc(&dC, sizeC);
        cudaMemcpy(dA, A_col.data(), sizeA, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, B_col.data(), sizeB, cudaMemcpyHostToDevice);
    }
    // Zero initialize C
    cudaMemset(dC, 0, (size_t)M * N * sizeof(float));

    // Prepare cuBLAS parameters
    const float alpha_f = 1.0f, beta_f = 0.0f;
    const __half alpha_h = __float2half(1.0f);
    const __half beta_h = __float2half(0.0f);

    float avg_gpu_time_s = 0.0f;
    for (int it = 0; it < repeat; ++it) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        if (use_fp16) {
            // inputs: dA (M x K) as FP16 column-major, dB (K x N) FP16 col-major
            // output: dC (M x N) FP32 column-major
            // use compute type CUBLAS_COMPUTE_32F_FAST_16F to use Tensor Cores
            cublasGemmEx(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha_f,
                dA, CUDA_R_16F, M,
                dB, CUDA_R_16F, K,
                &beta_f,
                dC, CUDA_R_32F, M,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
        } else {
            // FP32 path
            cublasSgemm(
                cublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &alpha_f,
                (const float*)dA, M,
                (const float*)dB, K,
                &beta_f,
                (float*)dC, M
            );
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        avg_gpu_time_s += ms * 1e-3f;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    avg_gpu_time_s /= (float)repeat;

    // Copy C back (column-major) and convert to row-major for writing
    std::vector<float> C_col((size_t)M * N);
    cudaMemcpy(C_col.data(), dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // convert col-major -> row-major
    std::vector<float> C_row((size_t)M * N);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C_row[(size_t)i * N + j] = C_col[(size_t)j * M + i];

    // write raw binary
    std::ofstream ofs("C_out.bin", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(C_row.data()), (size_t)M * N * sizeof(float));
    ofs.close();

    std::cout << "Average GPU time (s): " << avg_gpu_time_s << "\n";
    std::cout << "Wrote C_out.bin (raw float32 row-major). Use numpy: np.fromfile('C_out.bin',dtype=np.float32).reshape(("<<M<<","<<N<<"))\n";

    // cleanup
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(cublas);
    return 0;
}