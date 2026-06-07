#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <time.h>


float SCALE = 10000.;

__global__ void reduce_sum_v0(float* in_vec, float* out_vec, int vector_length) {
    if (threadIdx.x == 0) {
        out_vec[threadIdx.x] = 0;
        for (int i=0; i<vector_length; i++){
            out_vec[threadIdx.x] = out_vec[threadIdx.x] + in_vec[i];
        }
    } 
}

__global__ void reduce_sum_v1(float* in_vec, float* out_vec, int vector_length) {
    __shared__ float shared_data[1];
    if (threadIdx.x == 0) {
        out_vec[threadIdx.x] = 0;
        for (int i=0; i<vector_length; i++){
            shared_data[threadIdx.x] = shared_data[threadIdx.x] + in_vec[i];
        }
        out_vec[threadIdx.x] = shared_data[threadIdx.x];
    } 
}

__global__ void reduce_sum_v2(float* in_vec, float* out_vec, int vector_length) {
    float buffer = 0.0f;
    __shared__ float shared_data[1024];

    int steps = (vector_length + blockDim.x - 1) / blockDim.x;

    if (threadIdx.x == 0) {
        out_vec[threadIdx.x] = 0;
    };

    for (int step = 0; step < steps; step++) {
        int in_vec_idx = threadIdx.x + step * blockDim.x;
        if (in_vec_idx < vector_length) {
            shared_data[threadIdx.x] = in_vec[in_vec_idx];
        }
        else {
            shared_data[threadIdx.x] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            buffer = 0.0f;
            for (int i=0; i<blockDim.x; i++){
                buffer += shared_data[i];
            }
            out_vec[threadIdx.x] = out_vec[threadIdx.x] + buffer;
        }
        __syncthreads();
    }
}

__global__ void reduce_sum_v3(float* in_vec, float* out_vec, int vector_length) {
    __shared__ float shared_data[1024];
    float buffer = 0.0f;
    float local_sum = 0.0f;
    for (int i=threadIdx.x; i<vector_length; i+=blockDim.x){
        local_sum = local_sum + in_vec[i];
    }
    shared_data[threadIdx.x] = local_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i=0; i<blockDim.x; i++) {
            buffer += shared_data[i];
        }
        out_vec[threadIdx.x] = buffer;
    }
}

void init_array(float* in_vec, int length) {
    for (int i = 0; i < length; i++) {
        in_vec[i] = (float)i / SCALE;
    }
}

void serial_reduce_sum(float* in_vec, float* out_vec, int vector_length) {
    for (int i=0; i<vector_length; i++){
        out_vec[0] = out_vec[0] + in_vec[i];
    }
}

void compare_reduce_sums(int vector_length) {
    // const float real_answer = (vector_length * ((vector_length - 1)/ (2 * SCALE)));
    const double real_answer = ((double)vector_length * (((double)vector_length - 1.0) / (2.0 * SCALE)));
    printf("For vector length %d correct answer is: %f\n", vector_length, real_answer);

    struct timespec start, end;
    double elapsed;

    float* in_vec = nullptr;
    float* out_vec_cuda = nullptr;
    float* out_vec_serial = (float*) malloc(1 * sizeof(float));

    float* dev_in_vec = nullptr;
    float* dev_out_vec_cuda = nullptr;

    cudaMallocHost(&in_vec, vector_length * sizeof(float));
    cudaMallocHost(&out_vec_cuda, 1 * sizeof(float));
    cudaMalloc(&dev_in_vec, vector_length * sizeof(float));
    cudaMalloc(&dev_out_vec_cuda, 1 * sizeof(float));


    // Init arrays
    clock_gettime(CLOCK_MONOTONIC, &start);
    out_vec_cuda[0] = 0;
    init_array(in_vec, vector_length);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[CPU] Initialized array in %f second.\nArray[:5]: ", elapsed);
    for (int i=0; i<5; i++) {
        printf("%f ", in_vec[i]);
    }
    printf("\n");

    // Copy to CUDA
    clock_gettime(CLOCK_MONOTONIC, &start);
    cudaMemcpy(dev_in_vec, in_vec, vector_length * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(dev_out_vec_cuda, out_vec_cuda, 1 * sizeof(float), cudaMemcpyDefault);


    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[GPU] Copied array in %f second\n", elapsed);

    // Sum serial
    clock_gettime(CLOCK_MONOTONIC, &start);
    serial_reduce_sum(in_vec, out_vec_serial, vector_length);
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[CPU] Summed array in %f second. Result: %f.\n", elapsed, out_vec_serial[0]);
    
    // Sum GPU - 1
    clock_gettime(CLOCK_MONOTONIC, &start);
    reduce_sum_v0<<<1, 1>>>(dev_in_vec, dev_out_vec_cuda, vector_length);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cudaMemcpy(out_vec_cuda, dev_out_vec_cuda, 1 * sizeof(float), cudaMemcpyDefault);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[GPU] Summed array in %f second. Result: %f.\n", elapsed, out_vec_cuda[0]);

    // Sum GPU - 2
    clock_gettime(CLOCK_MONOTONIC, &start);
    reduce_sum_v1<<<1, 1>>>(dev_in_vec, dev_out_vec_cuda, vector_length);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cudaMemcpy(out_vec_cuda, dev_out_vec_cuda, 1 * sizeof(float), cudaMemcpyDefault);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[GPU] Summed array in %f second. Result: %f.\n", elapsed, out_vec_cuda[0]);

    // Sum GPU - 3
    clock_gettime(CLOCK_MONOTONIC, &start);
    reduce_sum_v2<<<1, 1024>>>(dev_in_vec, dev_out_vec_cuda, vector_length);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cudaMemcpy(out_vec_cuda, dev_out_vec_cuda, 1 * sizeof(float), cudaMemcpyDefault);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[GPU] Summed array in %f second. Result: %f.\n", elapsed, out_vec_cuda[0]);

    // Sum GPU - 4
    clock_gettime(CLOCK_MONOTONIC, &start);
    reduce_sum_v3<<<1, 1024>>>(dev_in_vec, dev_out_vec_cuda, vector_length);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    cudaMemcpy(out_vec_cuda, dev_out_vec_cuda, 1 * sizeof(float), cudaMemcpyDefault);
    elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("[GPU] Summed array in %f second. Result: %f.\n", elapsed, out_vec_cuda[0]);


    cudaFree(dev_in_vec);
    cudaFree(dev_out_vec_cuda);
    cudaFreeHost(in_vec);
    cudaFreeHost(out_vec_cuda);
    free(out_vec_serial);
}


int main(int argc, char**argv) {
    int vector_length = 1024;
    if(argc >=2)
    {
        vector_length = std::atoi(argv[1]);
    }
    compare_reduce_sums(vector_length);		
    return 0;
}