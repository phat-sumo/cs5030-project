#include <stdint.h>

// Device kernel equivalent of our CPU is_visible method. Intended to be called by other kernels
__device__ bool cuda_is_visible(int width, int height, short* d_values, int x0, int y0, int x1, int y1) {

	short elevation = d_values[width * x0 + y0];

	// The absolute difference between the x and y values we're going between.
	// dy is inverted to make some math simpler.
	int dx =  abs(x1 - x0);
	int dy = -abs(y1 - y0);

	// The step size, i.e. which direction we need to go from (x0, y0) to get to (x1, y1).
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;

	// The number of extra units we have to go to be perfectly aligned with the slope
	// If it's positive we lack more distance in the x direction.
	// If it's negative we lack more distance in the y direction.
	int err = dx + dy;

	while (true) {
		// if we find a taller point than us, we cannot see the destination
		if (d_values[width * x0 + y0] > elevation) {
			return false;
		}

		// we've reached our destination
		if (x0 == x1 && y0 == y1) {
			return true;
		}

		int e2 = err * 2;

		// should only be true if we do not need any distance in the y direction.
		if (e2 >= dy) {
			err += dy;
			x0 += sx;
		}

		// should only be true if we do not need any distance in the x direction.
		if (e2 <= dx) {
			err += dx;
			y0 += sy;
		}
	}
}

// Global kernel that gets an x, y coordinate and computes the bresenham line algorithm at that point
__global__ void cuda_bresenham(int width, int height, short* d_values, uint32_t* d_output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

	// Checks a 100 x 100 range area around coordinates (x, y) for visibility
    int sum = 0;
	for (int col = y - 100; col <= y + 100; col++) {
		if (col < 0 || col >= height) {
			continue;
        }

		for (int row = x - 100; row <= x + 100; row++) {
			if (row < 0 || row >= width) {
				continue;
			}

            if (cuda_is_visible(width, height, d_values, x, y, row, col)) {
				sum++;
			}
		}
	}
	d_output[width * y + x] = sum;
    //printf("Value computed for cell [%d, %d]\n", x, y);
}
