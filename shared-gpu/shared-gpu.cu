// Shared GPU implementation
#include <cassert>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "../common/cuda_bresenham.cu"

// Error check function given in gpu-extra.pdf
inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
    return result;
}

int main() {
	// Open file containing elevation data
	const char input_filename[] = "../common/srtm_14_04_6000x6000_short16.raw";
	FILE* input_file = fopen(input_filename, "r");

	if (input_file == NULL) {
		printf("could not open file %s\n", input_filename);
		return 1;
	}

	// Initialize map data
	const int width = 6000;
	const int height = 6000;
	const int num_values = width * height;

	// Read in elevation data
	short* h_values = (short*) malloc(num_values * sizeof(short));
	fread(h_values, sizeof(short), num_values * sizeof(short), input_file);
	fclose(input_file);

	// Set all elements to 0
	uint32_t* h_output = (uint32_t*) malloc(num_values * sizeof(uint32_t));
	memset(h_output, 0, num_values * sizeof(uint32_t));

	// Allocate cuda device variables
    short* d_values = NULL;
	uint32_t* d_output = NULL;
    checkCuda(cudaMalloc((void **)&d_values, num_values * sizeof(short)));
	checkCuda(cudaMalloc((void **)&d_output, num_values * sizeof(uint32_t)));
    checkCuda(cudaMemcpy(d_values, h_values, num_values * sizeof(short), cudaMemcpyHostToDevice));

	// Begin execution timing
	struct timespec ts_start;
	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	// Initialize the grid and block sizes from the map width and height
    const dim3 grid_size(width / 16, height / 16, 1);
    const dim3 block_size(16, 16, 1);
    cuda_bresenham<<<grid_size, block_size>>>(width, height, d_values, d_output);

	// Synchronize and transfer the results from the device back to the host
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_output, d_output, num_values * sizeof(uint32_t), cudaMemcpyDeviceToHost));

	// End execution timing
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	printf("Total elapsed time: %ld\n", (ts_end.tv_sec - ts_start.tv_sec) * 1000000);

	// Write output data to file
	const char output_filename[] = "../common/srtm_14_04_out_6000x6000_uint32.raw";
	FILE* output_file = fopen(output_filename, "w");
	fwrite(h_output, sizeof(unsigned char), num_values * sizeof(uint32_t), output_file);
	fclose(output_file);

	// Clean up allocated variables
	free(h_values);
	free(h_output);
	cudaFree(d_values);
	cudaFree(d_output);

	return 0;
}
