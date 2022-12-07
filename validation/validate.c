// validate.c - validates images for cs5030-project
// reads in two images, checks if they're the same, reports an error if not

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <omp.h>

void validate(uint32_t* a, uint32_t* b, int size) {

	bool differ = false;

# pragma omp parallel for
	for (int i = 0; i < size; i++) {

		if (a[i] != b[i]) {
			differ = true;
		}
	}

	if (differ) {
		printf("files differ. validation failed\n");
	} else {
		printf("files do not differ. validation successful\n");
	}
}

void read_file(uint32_t* p, const char filename[]) {
	
	// handle file input
	FILE* input_file = fopen(filename, "r");

	if (input_file == NULL) {
		printf("could not open file %s\n", filename);
		return;
	}

	fread(p, sizeof(uint32_t), 6000 * 6000 * sizeof(uint32_t), input_file);
	fclose(input_file);
}

int main() {

	const int fullsize = 6000*6000;

	// read in serial result
	const char serial_filename[] = "../output/srtm_14_04_serial_out_6000x6000_uint32.raw";
	const char shared_cpu_filename[] = "../output/srtm_14_04_shared_cpu_out_6000x6000_uint32.raw";
	const char shared_gpu_filename[] = "../output/srtm_14_04_shared_gpu_out_6000x6000_uint32.raw";
	const char distributed_cpu_filename[] = "../output/srtm_14_04_distributed_cpu_out_6000x6000_uint32.raw";
	const char distributed_gpu_filename[] = "../output/srtm_14_04_distributed_gpu_out_6000x6000_uint32.raw";

	uint32_t* serial;
	serial = (uint32_t*) malloc(fullsize * sizeof(uint32_t));
	uint32_t* shared_cpu;
	shared_cpu = (uint32_t*) malloc(fullsize * sizeof(uint32_t));
	uint32_t* shared_gpu;
	shared_gpu = (uint32_t*) malloc(fullsize * sizeof(uint32_t));
	uint32_t* distributed_cpu;
	distributed_cpu = (uint32_t*) malloc(fullsize * sizeof(uint32_t));
	uint32_t* distributed_gpu;
	distributed_gpu = (uint32_t*) malloc(fullsize * sizeof(uint32_t));

	printf("validating 6000x6000 problem size...\n\n");

	read_file(serial, serial_filename);
	read_file(shared_cpu, shared_cpu_filename);
	read_file(shared_gpu, shared_gpu_filename);
	read_file(distributed_cpu, distributed_cpu_filename);
	read_file(distributed_gpu, distributed_gpu_filename);

	printf("validating shared_cpu\n");
	validate(serial, shared_cpu, fullsize);
	printf("validating shared_gpu\n");
	validate(serial, shared_gpu, fullsize);
	printf("validating distributed_cpu\n");
	validate(serial, distributed_cpu, fullsize);
	printf("validating distributed_gpu\n");
	validate(serial, distributed_gpu, fullsize);

	free(serial);
	free(shared_cpu);
	free(shared_gpu);
	free(distributed_cpu);
	free(distributed_gpu);

	const int smolsize = 300 * 300;

	const char serial_smol_filename[] = "../output/srtm_14_04_serial_out_300x300_uint32.raw";
	const char shared_cpu_smol_filename[] = "../output/srtm_14_04_shared_cpu_out_300x300_uint32.raw";
	const char shared_gpu_smol_filename[] = "../output/srtm_14_04_shared_gpu_out_300x300_uint32.raw";
	const char distributed_cpu_smol_filename[] = "../output/srtm_14_04_distributed_cpu_out_300x300_uint32.raw";
	const char distributed_gpu_smol_filename[] = "../output/srtm_14_04_distributed_gpu_out_300x300_uint32.raw";

	uint32_t* serial_smol;
	serial_smol = (uint32_t*) malloc(smolsize * sizeof(uint32_t));
	uint32_t* shared_cpu_smol;
	shared_cpu_smol = (uint32_t*) malloc(smolsize * sizeof(uint32_t));
	uint32_t* shared_gpu_smol;
	shared_gpu_smol = (uint32_t*) malloc(smolsize * sizeof(uint32_t));
	uint32_t* distributed_cpu_smol;
	distributed_cpu_smol = (uint32_t*) malloc(smolsize * sizeof(uint32_t));
	uint32_t* distributed_gpu_smol;
	distributed_gpu_smol = (uint32_t*) malloc(smolsize * sizeof(uint32_t));

	printf("validating 300x300 problem size...\n\n");

	read_file(serial_smol, serial_smol_filename);
	read_file(shared_cpu_smol, shared_cpu_smol_filename);
	read_file(shared_gpu_smol, shared_gpu_smol_filename);
	read_file(distributed_cpu_smol, distributed_cpu_smol_filename);
	read_file(distributed_gpu_smol, distributed_gpu_smol_filename);

	printf("validating shared_cpu\n");
	validate(serial_smol, shared_cpu_smol, smolsize);
	printf("validating shared_gpu\n");
	validate(serial_smol, shared_gpu_smol, smolsize);
	printf("validating distributed_cpu\n");
	validate(serial_smol, distributed_cpu_smol, smolsize);
	printf("validating distributed_gpu\n");
	validate(serial_smol, distributed_gpu_smol, smolsize);




	return 0;
}
