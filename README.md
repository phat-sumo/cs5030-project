# How to build

1. Enter the directory of the target implementation.
2. `make` then `make run`
   * This should run the program with the maximum number of threads your system
     supports; if this behavior is not desired, modify the Makefile.

# Descriptions

## serial

Pseudocode:

* Read the image from a file.
* For each pixel in the image:
  * Initialize `count` to 0.
  * For each pixel not more than 100 pixels from the source in either direction:
    * Find the Bresenham line between the source and this pixel.
    * If no pixel on has a greater value than this, increment `count`.
  * Save `count` to an output array.
    * Initialize `count` to 0.
* Write the output array to a file.

## shared-cpu

Pseudocode:

* Read the image from a file.
* For each row in the image:
  * Spawn a new thread with OpenMP. On each thread:
    * Initialize `count` to 0.
    * For every `n`th pixel in this row, starting at this thread's rank, where
      `n` is the number of threads, perform the Bresenham line test as above
      for the surrounding pixels within a 100 pixel radius.
* Write the output array to a file.

## shared-gpu

Pseudocode:

* Read the image from a file.
* Allocate CUDA variables to store the map values and output.
* Copy the map values from host memory into cuda memory.
* Call the cuda_bresenham kernel to compute the elevation results for the dataset.
* Copy the computed output from cuda memory into host memory.
* Write the output array to a file.

## distributed-cpu

Psuedocode:

* On rank 0: Read the image from a file.
* On rank 0: Create `comm_size` equal slices of the image.
  * To this range, add enough surrounding pixels on either side so that the 100
    pixel radius for all pixels is satisfied; this will be `width * 100 + 100`
    pixels.
    ![An explanatory image](visualizations/explanatory_image.png)
  * Distribute these slices to each MPI process.
* On each MPI process:
  * For each pixel in the image, starting at the beginning of this process's
    slice and ending at the end of this process's slice, perform the Bresenham
    line test as above for the surrounding pixels within a 100 pixel radius.
* On rank 0:
  * Collect all of the worker processes' output slices and combine them into an
    output array.
  * Write the output array to a file.

## distributed-gpu

Psuedocode:

* On rank 0: Read the image from a file.
* On rank 0: Create `comm_size` equal slices of the image.
  * To this range, add enough surrounding pixels on either side so that the 100
    pixel radius for all pixels is satisfied; this will be `width * 100 + 100`
    pixels.
  * Distribute these slices to each MPI process.
* On each MPI process:
  * Allocate CUDA variables to store the map slice and output.
  * Copy the map slice from host memory into cuda memory.
  * Call the cuda_bresenham kernel to compute the elevation results for the dataset.
  * Copy the computed output from cuda memory into host memory.
* On rank 0:
  * Collect all of the worker processes' output slices and combine them into an
    output array.
  * Write the output array to a file.

# Performance study

## cpu comparison
We can see from this graph that the shared-cpu OpenMP implementation is
considerably faster than the serial implementation, showing about a 5x speedup.
The distributed-cpu OpenMPI implementation is still faster than a serial approach,
but is slower than a shared-memory approach due to the message-passing memory
overhead.

![cpu-timing](visualizations/cpu-implementation-timing)

We were able to compile and run our serial and shared-cpu implementations on the
CHPC's Notchpeak cluster. The node we selected had two 40-core Intel XeonSP
Cascadelake processors and 192GB of memory, although our approach did not heavily
utilize the expanded memory pool. The serial approach took a whopping 63546.402s
to complete, nearly 18 hours. The 80-core OpenMP implementation knocked that down
to 4311.036s, or just over an hour. The result is a 14.74x speedup.

We've also extrapolated a point in the graph to show about how we'd expect an
OpenMPI implementation to perform on the same node. Of course, the major benefits
of an OpenMPI approach would be shown as you scale the processing power up to
include multiple nodes, exceeding our previous 80-core limit.

## gpu comparison
The primary things we tested here were the effect that different grid and data sizes have on the execution time of our code.

![shared-gpu-timing](visualizations/shared-gpu-timing-300.png)

![shared-gpu-timing](visualizations/shared-gpu-timing-6000.png)

As can be seen, the optimal grid size was found to be between 8 and 16 for both the 300x300 and 6000x6000 datasets. Using a grid size of 4 results in a large performance loss at almost 2x the execution time compared to 8. Grid sizes of 32 performed marginally worse than 8-16, while grid sizes larger than 32 resulted in failed kernel launches. Our best guess as to why this is the case: grid sizes that are too small result in a small number of work to perform per grid. Grid sizes that are too large result in fewer grids, but too much work per grid.

# Visualizations

To visualize the raw files, including input as and results, we used both ImageJ
and Paraview. What follows is a JPEG version of the input file, exported from
ImageJ:

![The input file](visualizations/6000x6000_input.jpg)

Once we produced an output file from the serial implementation we also created
a JPEG export of that. The outputs from the parallel implementations looked
exactly the same, predictably, so the following image can be considered our
generalized output:

![The output file](visualizations/6000x6000_output.jpg)

During the course of testing, we took a 300x300 sample of the dataset for the
purposes of quicker testing. Here's that file:

![The short input file](visualizations/300x300_input.jpg)

And here's the output it produced, which was very similar to the portion of the
output that it rightly should have been:

![The short output file](visualizations/300x300_output.jpg)

We also wanted to get a side-by-side of the input and the output. For that
purpose we used Paraview to produce this nice visualization:

![Paraview my beloved](visualizations/6000x6000_paraview.png)
