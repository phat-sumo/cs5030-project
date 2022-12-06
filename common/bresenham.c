// Bresenham implementation
#include <stdbool.h>
#include <stdlib.h>
#include "bresenham.h"

// Returns true if the considered cell is visible from the origin cell
// Source: ADD SOURCE HERE
bool is_visible(ElevationMap map, int x0, int y0, int x1, int y1) {
	short elevation = map.values[map.width * x0 + y0];

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
		if (map.values[map.width * x0 + y0] > elevation) {
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

// Compute the partial map bounds for a given rank
void get_bounds(ElevationMap map, int comm_size, int rank, Bounds *bounds) {
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