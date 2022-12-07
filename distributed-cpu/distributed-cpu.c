// Distributed CPU implementation
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "../common/bresenham.h"

#define CLOCK_MONOTONIC 1

// Compute a partial elevation map given a start and end index
void fill_map(ElevationMap map, uint32_t* output, int startidx, int endidx) {
	int starty = startidx / map.height;
	int startx = startidx % map.height;

	// Compute the partial viewshed from elevation map
	for (int j = starty; j < map.height; j++) {
		for (int i = startx; i < map.width; i++) {
			int idx = map.width * j + i;
			if (idx > endidx) {
				return;
			}
			uint32_t sum = 0; 
			for (int y = j - 100; y <= j + 100; y++) {
				if (y < 0 || y >= map.height) {
					continue;
				}

				for (int x = i - 100; x <= i + 100; x++) {
					if (x < 0 || x >= map.width) {
						continue;
					}

					// If the considered cell is visible from the origin cell, increment sum
					if (is_visible(map, i, j, x, y)) {
						sum++;
					}	
				}
			}
			output[idx - startidx] = sum;
		}
		startx = 0;
		printf("Row %4d complete\n", j);
	}
}

int main(int argc, char* argv[]) {
	// Process the 6000x6000 data by default
	char input_filename[] = "../common/srtm_14_04_6000x6000_short16.raw";
	char output_filename[] = "../output/srtm_14_04_out_6000x6000_uint32.raw";
	/* char input_filename[] = "../common/srtm_14_04_300x300_short16.raw"; */
	/* char output_filename[] = "../output/srtm_14_04_distributed_cpu_out_300x300_uint32.raw"; */

	const int width = 6000;
	const int height = 6000;
	/* const int width = 300; */
	/* const int height = 300; */

	// Initialize MPI with size and rank
	MPI_Init(&argc, &argv);
	int comm_size;
	int my_rank;

	// Initialize MPI size and rank
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_size(comm, &comm_size);
	MPI_Comm_rank(comm, &my_rank);

	// Instantiate a map struct
	ElevationMap map;
	map.width = width;
	map.height = height;
	const int map_size = map.width * map.height;
	map.values = (short*) malloc(map_size * sizeof(short));

	// Instantiate a bounds struct
	Bounds bounds_local;
	get_bounds(map, comm_size, my_rank, &bounds_local);

	// Allocate a differently sized output depending on the rank
	uint32_t* output = NULL;
	if (my_rank == 0) {
		// Open file containing elevation data
		FILE* input_file = fopen(input_filename, "r");
		if (input_file == NULL) {
			printf("could not open file %s\n", input_filename);
			return 1;
		}

		// Read in elevation data
		fread(map.values, sizeof(short), map_size * sizeof(short), input_file);
		fclose(input_file);

		// Set all output elements to 0
		output = (uint32_t*) malloc(map_size * sizeof(uint32_t));
		memset(output, 0, map_size * sizeof(uint32_t));

		// Begin execution timing
		struct timespec ts_start;
		clock_gettime(CLOCK_MONOTONIC, &ts_start);

		// Distribute elevation data across MPI processes
		for (int rank = 1; rank < comm_size; rank++) {
			Bounds b;
			// get_bounds(map, comm_size, rank, &b);
			// MPI_Send(map.values + b.start, b.length, MPI_SHORT, rank, 0, comm);
			MPI_Send(map.values, map_size, MPI_SHORT, rank, 0, comm);
		}

		// Compute partial viewshed from elevation map
		fill_map(map, output, bounds_local.offset, bounds_local.offset + bounds_local.slice_size);

		// Wait to recieve all partial outputs from other processes
		for (int rank = 1; rank < comm_size; rank++) {
			Bounds b;
			get_bounds(map, comm_size, rank, &b);
			MPI_Recv(output + b.offset, b.slice_size, MPI_UINT32_T, rank, 1, comm, MPI_STATUS_IGNORE);
		}

		// End execution timing
		struct timespec ts_end;
		clock_gettime(CLOCK_MONOTONIC, &ts_end);
		printf("Total elapsed time: %ld\n", (ts_end.tv_sec - ts_start.tv_sec));

		uint32_t* out = (uint32_t*) malloc(map_size * sizeof(uint32_t));

		for (int i = 0; i < map.width; i++) {
			for (int j = 0; j < map.height; j++) {
				out[map.width * i + j] = output[map.height * j + i];
			}
		}

		// Write output data to file
		FILE* output_file = fopen(output_filename, "w");
		fwrite(out, sizeof(unsigned char), map_size * sizeof(uint32_t), output_file);
		fclose(output_file);
		free(out);
	} else {
		// Allocate partial output
		output = (uint32_t *) malloc(sizeof(uint32_t) * bounds_local.slice_size);

		// Recieve partial map values from rank 0
		// MPI_Recv(map.values + bounds_local.start, bounds_local.length, MPI_SHORT, 0, 0, comm, MPI_STATUS_IGNORE);
		MPI_Recv(map.values, map_size, MPI_SHORT, 0, 0, comm, MPI_STATUS_IGNORE);
		printf("%d: received message\n", my_rank);

		// Compute partial viewshed from elevation map
		fill_map(map, output, bounds_local.offset, bounds_local.offset + bounds_local.slice_size);

		// Send computed results back to rank 0
		MPI_Send(output, bounds_local.slice_size, MPI_UINT32_T, 0, 1, comm);
	}

	// Wait for all processes before ending the program
	MPI_Barrier(comm);	

	// Clean up allocated variables
	free(map.values);
	free(output);

  	// Clean up MPI
	MPI_Finalize();

	return 0;
}
