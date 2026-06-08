#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <time.h>

#define INDX(row, col, ld) ( ((row) * (ld)) + (col) )
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

void init_array(float* in_mat, int length) {
    for (int i = 0; i < length; i++) {
        in_mat[i] = (float)i;
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

void transpose_serial(float* in_mat, float* out_mat, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            out_mat[INDX(col, row, rows)] = in_mat[INDX(row, col, cols)];
        }
    }
}

__global__ void transpose_v0(float* in_mat, float* out_mat, int rows, int cols) {
    int source_x = blockIdx.x * blockDim.x + threadIdx.x;
    int source_y = blockIdx.y * blockDim.y + threadIdx.y;
    int source_idx = INDX(source_x, source_y, cols);
    int target_idx = INDX(source_y, source_x, rows);
    if (source_x < rows && source_y < cols){
        out_mat[target_idx] = in_mat[source_idx];
    }
}

// __global__ void transpose_v1(float* in_mat, float* out_mat, int rows, int cols) {
//     int source_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int source_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int source_idx = INDX(source_x, source_y, cols);
//     int target_idx = INDX(source_y, source_x, rows);
//     if (source_x < rows && source_y < cols){
//         out_mat[target_idx] = in_mat[source_idx];
//     }
// }

__global__ void transpose_v1(float* in_mat, float* out_mat, int rows, int cols) {
    int source_col = blockIdx.x * blockDim.x + threadIdx.x;
    int source_row = blockIdx.y * blockDim.y + threadIdx.y;
    int source_idx = INDX(source_row, source_col, cols);

    int target_col = blockIdx.y * blockDim.y + threadIdx.x;
    int target_row = blockIdx.x * blockDim.x + threadIdx.y;
    int target_idx = INDX(target_row, target_col, rows);
    __shared__ float buffer[32 * 32];

    if (source_row < rows && source_col < cols){
        buffer[threadIdx.y + threadIdx.x * blockDim.x] = in_mat[source_idx];
    }
    __syncthreads();
    if (target_row < cols && target_col < rows){
        out_mat[target_idx] = buffer[threadIdx.x + threadIdx.y * blockDim.y];
    }
}


__global__ void transpose_v2(float* in_mat, float* out_mat, int rows, int cols) {
    int source_col = blockIdx.x * blockDim.x + threadIdx.x;
    int source_row = blockIdx.y * blockDim.y + threadIdx.y;
    int source_idx = INDX(source_row, source_col, cols);

    int target_col = blockIdx.y * blockDim.y + threadIdx.x;
    int target_row = blockIdx.x * blockDim.x + threadIdx.y;
    int target_idx = INDX(target_row, target_col, rows);
    __shared__ float buffer[32][33];

    if (source_row < rows && source_col < cols){
        buffer[threadIdx.y][threadIdx.x] = in_mat[source_idx];
    }
    __syncthreads();
    if (target_row < cols && target_col < rows){
        out_mat[target_idx] = buffer[threadIdx.x][threadIdx.y];
    }
}

int check_mats_same(float* mat_a, float* mat_b, int rows, int cols) {
    float val_a;
    float val_b;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            val_a = mat_a[INDX(row, col, cols)];
            val_b = mat_b[INDX(row, col, cols)];
            if (val_a != val_b){
                printf("Matrices are not same. For col %d, row %d the values are %2.f and %2.f\n", row, col, val_a, val_b);
                return 0;
            }
        }
    }
    printf("Matrices are same.\n");
    return 0;
}

void compare_transposes(int rows, int cols) {
    struct timespec start, end;
    double elapsed;

    float* in_mat = nullptr;
    float* out_mat_cuda = nullptr;
    float* out_mat_serial = (float*) malloc(rows * cols * sizeof(float));

    float* dev_in_mat = nullptr;
    float* dev_out_mat_cuda = nullptr;

    cudaMallocHost(&in_mat, cols * rows * sizeof(float));
    cudaMallocHost(&out_mat_cuda, cols * rows * sizeof(float));
    cudaMalloc(&dev_in_mat, cols * rows * sizeof(float));
    cudaMalloc(&dev_out_mat_cuda, cols * rows * sizeof(float));


    clock_gettime(CLOCK_MONOTONIC, &start);
    init_array(in_mat, cols * rows);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    print_array(in_mat, rows, cols);
    printf("[CPU] Initialized matrix in %f seconds.\n\n", elapsed);

    
    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_serial(in_mat, out_mat_serial, rows, cols);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    print_array(out_mat_serial, cols, rows);
    printf("[CPU] Transposed matrix in %f seconds.\n\n", elapsed);


    cudaMemcpy(dev_in_mat, in_mat, rows * cols * sizeof(float), cudaMemcpyDefault);

    int threads_x = rows;
    int threads_y = cols;
    int grid_x = 1;
    int grid_y = 1;

    if (rows * cols > 1024) {
        threads_x = 32;
        threads_y = 32;
        grid_x = (rows + threads_x - 1) / threads_x;
        grid_y = (cols + threads_y - 1) / threads_y;

    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_v0<<<dim3(grid_x, grid_y),dim3(threads_x,threads_y)>>>(dev_in_mat, dev_out_mat_cuda, rows, cols);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, cols, rows);
    printf("[GPU] Transposed matrix in %f seconds.\n", elapsed);

    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols);
    printf("\n");


    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_v1<<<dim3(grid_y, grid_x),dim3(threads_x,threads_y)>>>(dev_in_mat, dev_out_mat_cuda, rows, cols);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, cols, rows);
    printf("[GPU] Transposed matrix in %f seconds.\n", elapsed);

    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols);
    printf("\n");



    clock_gettime(CLOCK_MONOTONIC, &start);
    transpose_v2<<<dim3(grid_y, grid_x),dim3(threads_x,threads_y)>>>(dev_in_mat, dev_out_mat_cuda, rows, cols);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    cudaMemcpy(out_mat_cuda, dev_out_mat_cuda, rows * cols * sizeof(float), cudaMemcpyDefault);
    print_array(out_mat_cuda, cols, rows);
    printf("[GPU] Transposed matrix in %f seconds.\n", elapsed);

    check_mats_same(out_mat_serial, out_mat_cuda, rows, cols);
    printf("\n");



    cudaFreeHost(in_mat);
    cudaFreeHost(out_mat_cuda);
    cudaFree(dev_in_mat);
    cudaFree(dev_out_mat_cuda);
    free(out_mat_serial);
}

int main(int argc, char** argv){
    int rows = 4;
    int cols = 4;
    if(argc >=2) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
    }
    compare_transposes(rows, cols);
    return 0;
}