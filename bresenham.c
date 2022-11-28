#include <stdbool.h>
#include <stdlib.h>

void plot_line(int x0, int y0, int x1, int y1, void callback(int x, int y)) {
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
		// this is the function that gets called for every pixel
		callback(x0, y0);

		// we've reached our destination
		if (x0 == x1 && y0 == y1) {
			break;
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
