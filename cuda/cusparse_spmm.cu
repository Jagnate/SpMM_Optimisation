// cusparse_spmm_bin.cu
// Read raw CSR binary arrays (A_data.bin float32, A_indices.bin int32, A_indptr.bin int32)
// and B.bin (float32 row-major). Run cusparseSpMM (FP32) and write C_out.bin (float32 row-major).
//
// Build:
//   nvcc -O2 cusparse_spmm_bin.cu -lcusparse -o cusparse_spmm_bin
//
// Run example:
//   ./cusparse_spmm_bin \
//     --adata case_out/A_data.bin --aind case_out/A_indices.bin --aindptr case_out/A_indptr.bin \
//     --b case_out/B.bin --m 1024 --k 1024 --n 1024 --out results/C_out_spmm.bin --repeat 3
//

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>

#include <cuda_runtime.h>
#include <cusparse.h>

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA ERROR %s:%d : %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

#define CHECK_CUSPARSE(x) do { cusparseStatus_t s = (x); if (s != CUSPARSE_STATUS_SUCCESS) { \
    fprintf(stderr, "CUSPARSE ERROR %s:%d : %d\n", __FILE__, __LINE__, (int)s); exit(1);} } while(0)

// read raw float32 binary (count floats)
bool read_raw_f32(const std::string &path, std::vector<float> &out, size_t count) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); return false; }
    out.resize(count);
    size_t r = fread(out.data(), sizeof(float), count, f);
    fclose(f);
    if (r != count) { fprintf(stderr, "read_raw_f32 expected %zu got %zu from %s\n", count, r, path.c_str()); return false; }
    return true;
}

// read raw int32 binary
bool read_raw_i32(const std::string &path, std::vector<int32_t> &out, size_t count) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); return false; }
    out.resize(count);
    size_t r = fread(out.data(), sizeof(int32_t), count, f);
    fclose(f);
    if (r != count) { fprintf(stderr, "read_raw_i32 expected %zu got %zu from %s\n", count, r, path.c_str()); return false; }
    return true;
}

bool write_raw_f32(const std::string &path, const std::vector<float> &data) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { perror(path.c_str()); return false; }
    size_t w = fwrite(data.data(), sizeof(float), data.size(), f);
    fclose(f);
    return w == data.size();
}

static void col_to_row_host(const float* col, float* row, int m, int n) {
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j)
            row[i * (size_t)n + j] = col[j * (size_t)m + i];
}

int main(int argc, char** argv) {
    std::string a_data_path, a_indices_path, a_indptr_path, b_bin_path, out_path = "C_out_spmm.bin";
    int M=0,K=0,N=0;
    int repeat=1;
    for (int i=1;i<argc;i++){
        std::string s = argv[i];
        if (s=="--adata") a_data_path = argv[++i];
        else if (s=="--aind") a_indices_path = argv[++i];
        else if (s=="--aindptr") a_indptr_path = argv[++i];
        else if (s=="--b") b_bin_path = argv[++i];
        else if (s=="--m") M = atoi(argv[++i]);
        else if (s=="--k") K = atoi(argv[++i]);
        else if (s=="--n") N = atoi(argv[++i]);
        else if (s=="--out") out_path = argv[++i];
        else if (s=="--repeat") repeat = atoi(argv[++i]);
    }

    if (a_data_path.empty() || a_indices_path.empty() || a_indptr_path.empty() || b_bin_path.empty() || M<=0 || K<=0 || N<=0) {
        fprintf(stderr, "Usage: %s --adata A_data.bin --aind A_indices.bin --aindptr A_indptr.bin --b B.bin --m M --k K --n N [--out C_out.bin] [--repeat R]\n", argv[0]);
        return 1;
    }

    // load A CSR arrays: we need nnz and m+1
    // To know counts: read A_indptr first to get m and nnz = indptr[m]-indptr[0]
    std::vector<int32_t> A_indptr;
    if (!read_raw_i32(a_indptr_path, A_indptr, (size_t)M + 1)) {
        fprintf(stderr, "Failed read A_indptr\n"); return 2;
    }
    size_t nnz = (size_t)A_indptr[M] - (size_t)A_indptr[0];
    if (A_indptr[0] != 0) {
        fprintf(stderr, "Warning: A_indptr[0] != 0 (is %d). This code assumes zero-based indexing.\n", A_indptr[0]);
    }

    std::vector<int32_t> A_indices;
    if (!read_raw_i32(a_indices_path, A_indices, nnz)) { fprintf(stderr, "Failed read A_indices\n"); return 2; }

    std::vector<float> A_data;
    if (!read_raw_f32(a_data_path, A_data, nnz)) { fprintf(stderr, "Failed read A_data\n"); return 2; }

    // load B row-major
    std::vector<float> B_row;
    if (!read_raw_f32(b_bin_path, B_row, (size_t)K * N)) { fprintf(stderr, "Failed read B.bin\n"); return 2; }

    // convert B to column-major for device
    std::vector<float> B_col((size_t)K * N);
    for (int i=0;i<K;++i)
        for (int j=0;j<N;++j)
            B_col[(size_t)j * K + i] = B_row[(size_t)i * N + j];

    // cuda / cusparse buffers
    CHECK_CUDA(cudaSetDevice(0));
    float *d_values = nullptr;
    int32_t *d_cols = nullptr;
    int32_t *d_rowptr = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_values, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_cols, nnz * sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc((void**)&d_rowptr, ((size_t)M + 1) * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_values, A_data.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cols, A_indices.data(), nnz * sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowptr, A_indptr.data(), ((size_t)M + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc((void**)&dB, (size_t)K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dC, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dB, B_col.data(), (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, (size_t)M * N * sizeof(float)));

    // create cusparse handle and descriptors
    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnMatDescr_t matB = nullptr;
    cusparseDnMatDescr_t matC = nullptr;

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, (int)nnz, d_rowptr, d_cols, d_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, K, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, M, dC, CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufSize));

    if (bufSize > 0) CHECK_CUDA(cudaMalloc(&dBuffer, bufSize));

    // warmup
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));
    CHECK_CUDA(cudaDeviceSynchronize());

    double avg_gpu_s = 0.0;
    for (int it=0; it<repeat; ++it) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start, 0));

        CHECK_CUSPARSE(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            &beta,
            matC,
            CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));

        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        avg_gpu_s += (double)ms * 1e-3;
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }
    avg_gpu_s /= (double)repeat;

    // copy C back col-major and convert to row-major
    std::vector<float> C_col((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(C_col.data(), dC, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> C_row((size_t)M * N);
    col_to_row_host(C_col.data(), C_row.data(), M, N);

    if (!write_raw_f32(out_path, C_row)) {
        fprintf(stderr, "Failed to write %s\n", out_path.c_str());
    }

    printf("cusparse SpMM average GPU time (s): %f\n", avg_gpu_s);
    printf("Wrote %s\n", out_path.c_str());

    // cleanup
    if (dBuffer) cudaFree(dBuffer);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);

    cudaFree(d_values);
    cudaFree(d_cols);
    cudaFree(d_rowptr);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}