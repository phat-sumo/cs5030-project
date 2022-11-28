#include <stdio.h>

#include "bresenham.h"

void print_coords(int x, int y) {
	printf("(%d, %d)\n", x, y);
}

int main() {
	plot_line(0, 0, 10, 4, &print_coords);
}
