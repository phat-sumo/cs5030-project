// Distributed GPU implementation
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <mpi.h>

#include "../common/bresenham.h"

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes used
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the current process rank
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Map map;
	map.width = 6000;
	map.height = 6000;
	const int num_map_cells = map.width * map.height;
    map.values = new short[num_map_cells];

    // Parse cli args (all processes do this, so that all processes have access to these values)
    if (rank == 0) {
        std::fstream infile;
        std::string input_file = "../common/srtm_14_04_6000x6000_short16.raw";
        infile.open(input_file, std::fstream::in | std::fstream::binary);
        infile.read((char *)&map.values[0], num_map_cells);
    }

    // Broadcast cli args values to all other processes
    MPI_Bcast(&map.values, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int *sendcounts = new int[size];
    int *displs = new int[size];

    // Compute the send and displacement amounts
    int sum = 0;
    int remainder = num_map_cells % size;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = num_map_cells / size;
        if (remainder > 0) {
            sendcounts[i] += 1;
            remainder -= 1;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }

    std::chrono::system_clock::time_point start = std::chrono::high_resolution_clock::now();

    // Reduce each local histogram into the final result and end execution timing
    // MPI_Reduce(local_bins, result, num_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    std::chrono::system_clock::time_point stop = std::chrono::high_resolution_clock::now();

    // Print results
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            std::cout << "displs[" << i << "] = " << displs[i] << std::endl;
        }
    }

    // Clean up and exit
    MPI_Finalize();
    return 0;
}