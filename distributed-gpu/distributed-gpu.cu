#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/cuda_bresenham.cu"

#define CLOCK_MONOTONIC 1

int main(int argc, char** argv) {
  // Initialize MPI
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Associate each MPI rank with a GPU device (assumes single node - multi gpu)
  int num_devices = 0;
  checkCuda(cudaGetDeviceCount(&num_devices));
  checkCuda(cudaSetDevice(rank % num_devices));
  checkCuda(cudaFree(0));

  char input_filename[] = "../common/srtm_14_04_6000x6000_short16.raw";
	char output_filename[] = "../common/srtm_14_04_out_6000x6000_uint32.raw";
	// char input_filename[] = "../common/srtm_14_04_300x300_short16.raw";
	// char output_filename[] = "../output/srtm_14_04_distributed_gpu_out_300x300_uint32.raw";

  const int width = 6000;
  const int height = 6000;
  // const int width = 300;
  // const int height = 300;

  ElevationMap map;
	map.width = width;
	map.height = height;
  const int map_size = map.width * map.height;
  map.values = (short*) malloc(map_size * sizeof(short));

  Bounds bounds_local;
	get_bounds(map, size, rank, &bounds_local);

  // Cuda device variables
  short* d_values = NULL;
	uint32_t* d_output = NULL;
  uint32_t* h_output = NULL;

  if (rank == 0) {
    // Open file containing elevation data
    FILE* input_file = fopen(input_filename, "r");

    if (input_file == NULL) {
		  printf("could not open file %s\n", input_filename);
		  return 1;
	  }

    // Read in elevation data
    fread(map.values, sizeof(short), map_size * sizeof(short), input_file);
    fclose(input_file);

    // Initialize the host's output memory
    h_output = (uint32_t*) malloc(map_size * sizeof(uint32_t));
		memset(h_output, 0, map_size * sizeof(uint32_t));

    // Distribute elevation data across MPI processes
		for (int rank = 1; rank < size; rank++) {
			Bounds b;
			get_bounds(map, size, rank, &b);
			MPI_Send(map.values + b.start, b.length, MPI_SHORT, rank, 0, MPI_COMM_WORLD);
		}

    checkCuda(cudaMalloc((void **)&d_values, bounds_local.slice_size * sizeof(short)));
	  checkCuda(cudaMalloc((void **)&d_output, map_size * sizeof(uint32_t)));
    checkCuda(cudaMemcpy(d_values, map.values, bounds_local.slice_size * sizeof(short), cudaMemcpyHostToDevice));

    struct timespec ts_start;
		clock_gettime(CLOCK_MONOTONIC, &ts_start);

    const dim3 grid_size(width / 16, height / 16, 1);
  	const dim3 block_size(16, 16, 1);
	  cuda_bresenham<<<grid_size, block_size>>>(width, height, d_values, d_output);

    // Synchronize and transfer the results from the device back to the host
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_output, d_output, bounds_local.slice_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		for (int rank = 1; rank < size; rank++) {
			Bounds b;
			get_bounds(map, size, rank, &b);
			MPI_Recv(h_output + b.offset, b.slice_size, MPI_UINT32_T, rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

    struct timespec ts_end;
		clock_gettime(CLOCK_MONOTONIC, &ts_end);

		printf("Total elapsed time: %ld\n", (ts_end.tv_sec - ts_start.tv_sec));

		// write data back to file
		FILE* output_file = fopen(output_filename, "w");
		fwrite(h_output, sizeof(unsigned char), map_size * sizeof(uint32_t), output_file);
		fclose(output_file);
  } else {
    h_output = (uint32_t *) malloc(sizeof(uint32_t) * bounds_local.slice_size);
    MPI_Recv(map.values + bounds_local.start, bounds_local.length, MPI_SHORT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf("%d: received message\n", rank);

    checkCuda(cudaMalloc((void **)&d_values, bounds_local.slice_size * sizeof(short)));
	  checkCuda(cudaMalloc((void **)&d_output, map_size * sizeof(uint32_t)));
    checkCuda(cudaMemcpy(d_values, map.values, bounds_local.slice_size * sizeof(short), cudaMemcpyHostToDevice));

    const dim3 grid_size(width / 16, height / 16, 1);
  	const dim3 block_size(16, 16, 1);
	  cuda_bresenham<<<grid_size, block_size>>>(width, height, d_values, d_output);

    // Synchronize and transfer the results from the device back to the host
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(h_output, d_output, bounds_local.slice_size * sizeof(uint32_t), cudaMemcpyDeviceToHost));

		MPI_Send(h_output, bounds_local.slice_size, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD);
  }

  // Clean up
  cudaFree(d_values);
  cudaFree(d_output);
  free(map.values);
  free(h_output);
  
  MPI_Finalize();

  return 0;
}
