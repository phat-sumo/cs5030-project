#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <omp.h>

#include "../common/bresenham.h"

int main() {

	char input_filename[] = "srtm_14_04_6000x6000_short16.raw";
	char output_filename[] = "srtm_14_04_out_6000x6000_uint32.raw";

	// handle file input
	FILE* input_file = fopen(input_filename, "r");

	if (input_file == NULL) {
		printf("could not open file %s\n", input_filename);
		return 1;
	}

	Map map;
	map.width = 6000;
	map.height = 6000;

	// read in our data
	map.values = (short*) malloc(map.width * map.height * sizeof(short));
	fread(map.values, sizeof(short), map.width * map.height * sizeof(short), input_file);

	fclose(input_file);

	// set up output array as 0s
	uint32_t* output = (uint32_t*) malloc(map.width * map.height * sizeof(uint32_t));
	memset(output, 0, map.width * map.height * sizeof(uint32_t));

	struct timespec ts_start;
	struct timespec ts_end;
	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	uint32_t sum = 0; 
	int i, j, x, y;
	int my_rank, thread_count;
	for (j = 0; j < map.height; j++) {
#		pragma omp parallel num_threads(4) private(y, x, sum, my_rank, thread_count)
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

						if (is_visible(map, i, j, x, y)) {
							/* printf("found [%d][%d] -> [%d][%d]\n", i, j, x, y); */
							sum++;
						}
							
					}
				}

				//printf("total [%d][%d]: %d\n", i, j, sum);
				output[map.width * j + i] = sum;
				
			}
		}
		printf("row %4d complete\n", j);
	}

	clock_gettime(CLOCK_MONOTONIC, &ts_end);

	printf("elapsed: %ld\n", (ts_end.tv_nsec - ts_start.tv_nsec) * 1000 * 1000);


	// write data back to file
	FILE* output_file = fopen(output_filename, "w");

	fwrite(output, sizeof(unsigned char), map.height * map.width * sizeof(uint32_t), output_file);

	fclose(output_file);

	// free what you malloc (or else) 
	free(map.values);
	free(output);

	return 0;
}
