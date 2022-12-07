// Shared CPU implemenation
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "../common/bresenham.h"

#define CLOCK_MONOTONIC 1

int main() {
	// Process the 6000x6000 data by default
	char input_filename[] = "../common/srtm_14_04_6000x6000_short16.raw";
	char output_filename[] = "../output/srtm_14_04_shared_cpu_out_6000x6000_uint32.raw";
	// char input_filename[] = "../common/srtm_14_04_300x300_short16.raw";
	// char output_filename[] = "../output/srtm_14_04_shared_cpu_out_300x300_uint32.raw";

	const int width = 6000;
	const int height = 6000;
	// const int width = 300;
	// const int height = 300;

	// Open file containing elevation data
	FILE* input_file = fopen(input_filename, "r");
	if (input_file == NULL) {
		printf("could not open file %s\n", input_filename);
		return 1;
	}

	// Instantiate a map struct
	ElevationMap map;
	map.width = width;
	map.height = height;

	// Read in elevation data
	map.values = (short*) malloc(map.width * map.height * sizeof(short));
	fread(map.values, sizeof(short), map.width * map.height * sizeof(short), input_file);
	fclose(input_file);

	// Set all output elements to 0
	uint32_t* output = (uint32_t*) malloc(map.width * map.height * sizeof(uint32_t));
	memset(output, 0, map.width * map.height * sizeof(uint32_t));

	// Begin execution timing
	struct timespec ts_start;
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	// Compute viewshed from elevation map
	uint32_t sum = 0; 
	int i, j, x, y;
	int my_rank, thread_count;
	for (j = 0; j < map.height; j++) {
		#pragma omp parallel private(y, x, sum, my_rank, thread_count)
		{
			my_rank = omp_get_thread_num();
			thread_count = omp_get_num_threads();
			for (i = my_rank; i < map.width; i += thread_count) {
				sum = 0;
				for (y = j - 100; y <= j + 100; y++) {
					if (y < 0 || y >= map.height) {
						continue;
					}

					for (x = i - 100; x <= i + 100; x++) {
						if (x < 0 || x >= map.width) {
							continue;
						}

						// If the considered cell is visible from the origin cell, increment sum
						if (is_visible(map, i, j, x, y)) {
							sum++;
						}	
					}
				}
				output[map.width * i + j] = sum;
			}
		}
		printf("row %4d complete\n", j);
	}

	// End execution timing
	clock_gettime(CLOCK_MONOTONIC, &ts_end);
	printf("elapsed: %lds\n", (ts_end.tv_sec - ts_start.tv_sec));

	// Write output data to file
	FILE* output_file = fopen(output_filename, "w");
	fwrite(output, sizeof(unsigned char), map.height * map.width * sizeof(uint32_t), output_file);
	fclose(output_file);

	// Clean up allocated variables
	free(map.values);
	free(output);

	return 0;
}
