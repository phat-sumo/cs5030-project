// Serial implementation
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <mpi.h>

#include "../common/bresenham.h"

typedef struct {
	int offset;
	int slice_size;
	int start;
	int length;
} Bounds;

void fill_map(Map map, uint32_t* output, int startidx) {
	int starty = startidx / map.height;
	int startx = startidx % map.height;

	for (int j = starty; j < map.height; j++) {
		for (int i = startx; i < map.width; i++) {
			uint32_t sum = 0; 

			for (int y = j - 100; y <= j + 100; y++) {
				if (y < 0 || y >= map.height) {
					continue;
				}

				for (int x = i - 100; x <= i + 100; x++) {
					if (x < 0 || x >= map.width) {
						continue;
					}

					if (is_visible(map, i, j, x, y)) {
						sum++;
					}
						
				}
			}

			//printf("total [%d][%d]: %d\n", i, j, sum);
			output[map.width * j + i - startidx] = sum;
			
		}

		startx = 0;
		printf("Row %4d complete\n", j);
	}
}

void get_bounds(Map map, int comm_size, int rank, Bounds *bounds) {
	int map_size = map.width * map.height;

	int normal_slice_length = map_size / comm_size;
	int slice_remainder = map_size % comm_size;
	// the number of extra pixels that we need on each side of a given selection
	int extra_pixels = 100 * map.width + 100;

	bounds->offset = normal_slice_length * rank + (rank < slice_remainder ? rank : slice_remainder);

	bounds->slice_size = normal_slice_length + (rank < slice_remainder ? 1 : 0);

	bounds->start = bounds->offset - extra_pixels;
	bounds->start = bounds->start < 0 ? 0 : bounds->start;

	bounds->length = bounds->slice_size + extra_pixels;
	bounds->length = bounds->offset + bounds->length > map_size ? map_size - bounds->offset : bounds->length;
}

int main(int argc, char* argv[]) {
	char input_filename[] = "../common/srtm_14_04_6000x6000_short16.raw";
	char output_filename[] = "../common/srtm_14_04_out_6000x6000_uint32.raw";

	MPI_Init(&argc, &argv);
	int comm_size;
	int my_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	Map map;
	map.width = 6000;
	map.height = 6000;

	const int map_size = map.width * map.height;
	map.values = (short*) malloc(map_size * sizeof(short));

	Bounds bounds_local;
	get_bounds(map, comm_size, my_rank, &bounds_local);

	// set up output array as 0s
	uint32_t* output;

	if (my_rank == 0) {
		// handle file input
		FILE* input_file = fopen(input_filename, "r");

		if (input_file == NULL) {
			printf("could not open file %s\n", input_filename);
			return 1;
		}

		// read in our data
		fread(map.values, sizeof(short), map_size * sizeof(short), input_file);

		fclose(input_file);

		output = (uint32_t*) malloc(map_size * sizeof(uint32_t));
		memset(output, 0, map_size * sizeof(uint32_t));

		struct timespec ts_start;
		clock_gettime(CLOCK_MONOTONIC, &ts_start);

		for (int rank = 1; rank < comm_size; rank++) {
			Bounds b;
			get_bounds(map, comm_size, rank, &b);

			MPI_Send(map.values + b.start, b.length, MPI_SHORT, rank, 0, MPI_COMM_WORLD);
		}

		fill_map(map, output, bounds_local.offset);

		for (int rank = 1; rank < comm_size; rank++) {
			Bounds b;
			get_bounds(map, comm_size, rank, &b);

			MPI_Status status;
			MPI_Recv(map.values + b.offset, b.slice_size, MPI_UINT32_T, rank, 1, MPI_COMM_WORLD, &status);
		}

		struct timespec ts_end;
		clock_gettime(CLOCK_MONOTONIC, &ts_end);

		printf("Total elapsed time: %ld\n", (ts_end.tv_sec - ts_start.tv_sec) * 1000000);

		// write data back to file
		FILE* output_file = fopen(output_filename, "w");
		fwrite(output, sizeof(unsigned char), map_size * sizeof(uint32_t), output_file);
		fclose(output_file);
	} else {
		output = (uint32_t *) malloc(sizeof(uint32_t) * bounds_local.slice_size);

		MPI_Status status;
		MPI_Recv(map.values + bounds_local.start, bounds_local.length, MPI_SHORT, 0, 0, MPI_COMM_WORLD, &status);
		printf("%d: received message\n", my_rank);

		fill_map(map, output, bounds_local.offset);

		MPI_Send(output, bounds_local.slice_size, MPI_UINT32_T, 0, 1, MPI_COMM_WORLD);
	}

	MPI_Finalize();

	// free what you malloc (or else) 
	free(map.values);
	free(output);

	return 0;
}
