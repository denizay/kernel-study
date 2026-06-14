#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define INDX(row, col, ld) ( ((row) * (ld)) + (col) )

// The single fastest and single slowest timed runs are dropped, leaving 8.
#define WARMUP_RUNS 1
#define TIMED_RUNS  10

void init_array(float* in_mat, int length) {
    for (int i = 0; i < length; i++) {
        in_mat[i] = (float)(i % 1235);
    }
}

void print_array(float* in_mat, int rows, int cols) {
    int ld = cols;
    if (rows > 5) {
        rows = 5;
    }
    if (cols > 5) {
        cols = 5;
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("%.0f ", in_mat[INDX(row, col, ld)]);
        }
        printf("\n");
    }
}

void serial_matmul(float* mat_left, float* mat_right, float* out_mat, int M, int N, int K){
    float sum;
    for (int m = 0; m < M; m++) {
        for (int k = 0; k < K; k++) {
            sum = 0;
            for (int n = 0; n < N; n++) {
                sum += mat_left[INDX(m, n, N)] * mat_right[INDX(n, k, K)]; 
            }
            out_mat[INDX(m, k, K)] = sum;
        }
    }
}

__global__ void matmul_v0(float* mat_left, float* mat_right, float* out_mat, int M, int N, int K) {
    int program_id = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row = program_id / K;
    int out_col = program_id - out_row * K;
    float sum = 0;
    if (out_row < M && out_col < K){
        for (int n = 0; n < N; n++) {
            sum += mat_left[INDX(out_row, n, N)] * mat_right[INDX(n, out_col, K)]; 
        }
            out_mat[INDX(out_row, out_col, K)] = sum;
    }
}

__global__ void matmul_v1(float* mat_left, float* mat_right, float* out_mat, int M, int N, int K) {
    int out_row =  blockIdx.y * blockDim.y + threadIdx.y;
    int out_col =  blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (out_row < M && out_col < K){
        for (int n = 0; n < N; n++) {
            sum += mat_left[INDX(out_row, n, N)] * mat_right[INDX(n, out_col, K)]; 
        }
            out_mat[INDX(out_row, out_col, K)] = sum;
    } 
}

__global__ void matmul_v2(float* mat_left, float* mat_right, float* out_mat, int M, int N, int K) {
    int out_row =  blockIdx.y * blockDim.y + threadIdx.y;
    int out_col =  blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float right_buffer[32][32];
    __shared__ float left_buffer[32][32];

    int repeats =  (N + 32 - 1)/32;
    float sum = 0;
    for (int repeat = 0; repeat < repeats; repeat ++){
        int load_right_row = repeat * 32 + threadIdx.y;
        int load_left_col = repeat * 32 + threadIdx.x;
        if (load_right_row < N) {
            right_buffer[threadIdx.y][threadIdx.x] = mat_right[INDX(load_right_row, out_col, K)];
        }
        else{
            right_buffer[threadIdx.y][threadIdx.x] = 0;
        }
        if (load_left_col < N) {
            left_buffer[threadIdx.y][threadIdx.x] = mat_left[INDX(out_row, load_left_col, N)];
        }
        else {
            left_buffer[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int n = 0; n < 32; n++) {
            sum += left_buffer[threadIdx.y][n] * right_buffer[n][threadIdx.x];
        }
        __syncthreads();
    }
    if (out_row < M && out_col < K){
        out_mat[INDX(out_row, out_col, K)] = sum;
    }
}

const int SIZE_RIGHT_BLOCK = 2;
const int SIZE_LEFT_BLOCK = 4;

__global__ void matmul_v3(float* mat_left, float* mat_right, float* out_mat, int M, int N, int K) {
    __shared__ float right_buffer[SIZE_RIGHT_BLOCK][32][32];
    __shared__ float left_buffer[SIZE_LEFT_BLOCK][32][32];

    int repeats =  (N + 32 - 1)/32;
    float sum[SIZE_LEFT_BLOCK][SIZE_RIGHT_BLOCK] = {};

    for (int repeat = 0; repeat < repeats; repeat ++){
        int load_right_row = repeat * 32 + threadIdx.y;
        if (load_right_row < N) {
            for (int i=0; i < SIZE_RIGHT_BLOCK; i++) {
                int out_col =  (blockIdx.x + (gridDim.x * i) ) * blockDim.x + threadIdx.x;
                if (out_col < K) {
                    right_buffer[i][threadIdx.y][threadIdx.x] = mat_right[INDX(load_right_row, out_col, K)];
                } else {
                    right_buffer[i][threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
        }
        
        int load_left_col = repeat * 32 + threadIdx.x;
        if (load_left_col < N) {
            for (int i=0; i < SIZE_LEFT_BLOCK; i++) {
                int out_row =  (blockIdx.y + (gridDim.y * i)) * blockDim.y + threadIdx.y;
                if (out_row < M) {
                    left_buffer[i][threadIdx.y][threadIdx.x] = mat_left[INDX(out_row, load_left_col, N)];
                } else {
                    left_buffer[i][threadIdx.y][threadIdx.x] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (int n = 0; n < 32; n++) {
            for (int i=0; i < SIZE_LEFT_BLOCK; i++) {
                for (int j=0; j < SIZE_RIGHT_BLOCK; j++) {
                    sum[i][j] += left_buffer[i][threadIdx.y][n] * right_buffer[j][n][threadIdx.x];
                }
            }
        }
        __syncthreads();
    }
    for (int i=0; i < SIZE_LEFT_BLOCK; i++) {
        int out_row =  (blockIdx.y + (gridDim.y * i)) * blockDim.y + threadIdx.y;
        if (out_row < M) {
            for (int j=0; j < SIZE_RIGHT_BLOCK; j++) {
                int out_col =  (blockIdx.x + (gridDim.x * j) ) * blockDim.x + threadIdx.x;
                if (out_col < K) {
                    out_mat[INDX(out_row, out_col, K)] = sum[i][j];
                }
            }
        }
    }
}


int check_mats_same(float* mat_a, float* mat_b, int rows, int cols,
                    float rel_tol, float abs_tol) {
    int    mismatches      = 0;
    double max_abs_err     = 0.0;
    double max_rel_err     = 0.0;
    int    worst_row       = -1;
    int    worst_col       = -1;
    float  worst_a         = 0.0f;
    float  worst_b         = 0.0f;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            float val_a = mat_a[INDX(row, col, cols)];
            float val_b = mat_b[INDX(row, col, cols)];

            double abs_err = fabs((double)val_a - (double)val_b);
            double scale   = fmax(fabs((double)val_a), fabs((double)val_b));
            double rel_err = (scale > 0.0) ? abs_err / scale : 0.0;

            // Track the single worst element by relative error.
            if (rel_err > max_rel_err) {
                max_rel_err = rel_err;
                worst_row   = row;
                worst_col   = col;
                worst_a     = val_a;
                worst_b     = val_b;
            }
            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
            }

            double allowed = fmax((double)rel_tol * scale, (double)abs_tol);
            if (abs_err > allowed) {
                mismatches++;
            }
        }
    }

    int total = rows * cols;
    if (mismatches == 0) {
        printf("Matrices match within tolerance "
               "(rel_tol=%.1e, abs_tol=%.1e).\n", rel_tol, abs_tol);
        printf("  max abs error: %.6g   max rel error: %.6g\n",
               max_abs_err, max_rel_err);
        return 1;
    } else {
        printf("Matrices DIFFER beyond tolerance "
               "(rel_tol=%.1e, abs_tol=%.1e).\n", rel_tol, abs_tol);
        printf("  mismatched elements: %d / %d (%.2f%%)\n",
               mismatches, total, 100.0 * mismatches / total);
        printf("  max abs error: %.6g   max rel error: %.6g\n",
               max_abs_err, max_rel_err);
        if (worst_row >= 0) {
            printf("  worst element at row %d, col %d: %.6g vs %.6g "
                   "(abs %.6g, rel %.6g)\n",
                   worst_row, worst_col, worst_a, worst_b,
                   fabs((double)worst_a - (double)worst_b), max_rel_err);
        }
        return 0;
    }
}

template <typename LaunchFn>
double benchmark_kernel(const char* name, LaunchFn launch,
                        int warmup_runs, int timed_runs) {
    struct timespec start, end;

    // Warmup (untimed).
    for (int i = 0; i < warmup_runs; i++) {
        launch();
    }
    cudaDeviceSynchronize();

    std::vector<double> times;
    times.reserve(timed_runs);
    for (int i = 0; i < timed_runs; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        launch();
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;
        times.push_back(elapsed);
    }

    std::sort(times.begin(), times.end());

    double avg;
    int kept;
    if ((int)times.size() > 2) {
        // Drop fastest (front) and slowest (back), average the rest.
        double sum = 0.0;
        for (size_t i = 1; i + 1 < times.size(); i++) {
            sum += times[i];
        }
        kept = (int)times.size() - 2;
        avg  = sum / kept;
    } else {
        double sum = 0.0;
        for (size_t i = 0; i < times.size(); i++) sum += times[i];
        kept = (int)times.size();
        avg  = sum / kept;
    }

    printf("[GPU %s] %d warmup + %d timed runs "
           "(dropped fastest %f s and slowest %f s)\n",
           name, warmup_runs, timed_runs, times.front(), times.back());
    printf("[GPU %s] avg over remaining %d runs: %f seconds.\n\n",
           name, kept, avg);

    return avg;
}

void compare_matmul(int rows, int cols, int cols_2) {
    struct timespec start, end;
    double elapsed;
    
    float* in_mat_left = nullptr;
    float* in_mat_right = nullptr;
    float* out_mat_cuda = nullptr;
    float* out_mat_serial = (float*) malloc(rows * cols_2 * sizeof(float));

    float* dev_in_mat_left = nullptr;
    float* dev_in_mat_right = nullptr;
    float* dev_out_mat_cuda = nullptr;

    cudaMallocHost(&in_mat_left, rows * cols * sizeof(float));
    cudaMallocHost(&in_mat_right, cols * cols_2 * sizeof(float));
    cudaMallocHost(&out_mat_cuda, rows * cols_2 * sizeof(float));

    cudaMalloc(&dev_in_mat_left, rows * cols * sizeof(float));
    cudaMalloc(&dev_in_mat_right, cols * cols_2 * sizeof(float));
    cudaMalloc(&dev_out_mat_cuda, rows * cols_2 * sizeof(float));

    // INIT
    clock_gettime(CLOCK_MONOTONIC, &start);
    init_array(in_mat_left, rows * cols);
    init_array(in_mat_right, cols * cols_2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[CPU] Initialized matrix in %f seconds.\n\n", elapsed);
    print_array(in_mat_left, rows, cols);
    printf("\n");
    print_array(in_mat_right, cols, cols_2);
    printf("\n");

    cudaMemcpy(dev_in_mat_left, in_mat_left, rows * cols * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(dev_in_mat_right, in_mat_right, cols * cols_2 * sizeof(float), cudaMemcpyDefault);


    // CPU (single run)
    clock_gettime(CLOCK_MONOTONIC, &start);
    serial_matmul(in_mat_left, in_mat_right, out_mat_serial, rows, cols, cols_2);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[CPU] MatMul matrix in %f seconds.\n\n", elapsed);
    print_array(out_mat_serial, rows, cols_2);
    printf("\n");


    // GPU launch configuration (shared by v1 and v2)
    int block_dim_x = 32;
    int block_dim_y = 32;
    dim3 blockDim(block_dim_x, block_dim_y, 1);
    int grid_dim_x = ((cols_2 + block_dim_x - 1) / block_dim_x)/ SIZE_RIGHT_BLOCK;
    int grid_dim_y = ((rows + block_dim_y - 1) / block_dim_y )/ SIZE_LEFT_BLOCK;
    dim3 gridDim(grid_dim_x, grid_dim_y, 1);

    printf("grid_dim_y: %d\n", grid_dim_y);

    // GPU V3
    benchmark_kernel("v3", [&]() {
        matmul_v3<<<gridDim, blockDim>>>(dev_in_mat_left, dev_in_mat_right,
                                         dev_out_mat_cuda, rows, cols, cols_2);
    }, WARMUP_RUNS, TIMED_RUNS);
    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols_2 * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, rows, cols_2);
    printf("\n");
    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols_2, 1e-4f, 1e-3f);
    printf("\n");

    // GPU launch configuration (shared by v1 and v2)
    block_dim_x = 32;
    
    block_dim_y = 32;
    blockDim = dim3(block_dim_x, block_dim_y, 1);
    grid_dim_x = (cols_2 + block_dim_x - 1) / block_dim_x;
    grid_dim_y = (rows + block_dim_y - 1) / block_dim_y;
    gridDim = dim3(grid_dim_x, grid_dim_y, 1);

    // GPU V2
    benchmark_kernel("v2", [&]() {
        matmul_v2<<<gridDim, blockDim>>>(dev_in_mat_left, dev_in_mat_right,
                                         dev_out_mat_cuda, rows, cols, cols_2);
    }, WARMUP_RUNS, TIMED_RUNS);
    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols_2 * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, rows, cols_2);
    printf("\n");
    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols_2, 1e-4f, 1e-3f);
    printf("\n");

    // GPU V1
    benchmark_kernel("v1", [&]() {
        matmul_v1<<<gridDim, blockDim>>>(dev_in_mat_left, dev_in_mat_right,
                                         dev_out_mat_cuda, rows, cols, cols_2);
    }, WARMUP_RUNS, TIMED_RUNS);
    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols_2 * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, rows, cols_2);
    printf("\n");
    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols_2, 1e-4f, 1e-3f);
    printf("\n");

    // GPU V0
    int thread_count = 1024;
    int num_blocks = ((rows * cols_2) + thread_count - 1) / thread_count;
    benchmark_kernel("v0", [&]() {
        matmul_v0<<<num_blocks, thread_count>>>(dev_in_mat_left, dev_in_mat_right,
                                                dev_out_mat_cuda, rows, cols, cols_2);
    }, WARMUP_RUNS, TIMED_RUNS);
    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols_2 * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, rows, cols_2);
    printf("\n");
    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols_2, 1e-4f, 1e-3f);

    // CLEAN
    cudaFreeHost(in_mat_left);
    cudaFreeHost(in_mat_right);
    cudaFreeHost(out_mat_cuda);
    cudaFree(dev_in_mat_left);
    cudaFree(dev_in_mat_right);
    cudaFree(dev_out_mat_cuda);
    free(out_mat_serial);
}

int main(int argc, char** argv){
    int rows = 1024;
    int cols = 1024;
    int cols_2 = 1024;
    if(argc >=2) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        cols_2 = std::atoi(argv[3]);
    }
    compare_matmul(rows, cols, cols_2);
    return 0;
}