#ifndef _KERNEL_HEADERS_
#define _KERNEL_HEADERS_

#include <stdint.h>
#include <iostream>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Header files for image read/write
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

using namespace std;

#define NUM_BINS 256
#define BLOCK_SIZE 32//32//16//32//16//32
#define MASK_SIZE 3//7//5//5//5//3//3
#define MASK_DIM 9//49//25//25//25//9//9
#define TILE_SIZE 30//26//28//10//28//14//30
#define LOW_TILE_SIZE 28//20//24//4//24//12//28

#define imax(a,b) (a > b) ? a : b;
#define imin(a,b) (a < b) ? a : b;

__constant__ float mask_nr[MASK_DIM];
__constant__ uint8_t mask_de[MASK_DIM];

__global__ void rgb2gray_invesion(uint8_t*, uint8_t*, int, int);
__global__ void noise_removal(uint8_t*, uint8_t*, float*, int, int);
__global__ void tiled_noise_removal(uint8_t*, uint8_t*, int, int);
__global__ void constant_memory_noise_removal(uint8_t*, uint8_t*, int, int);
__global__ void tiled_rgb2gray_inversion_noise_removal_binarization(uint8_t*, uint8_t*, int, int);

__global__ void naive_binarization(uint8_t*, uint8_t*, int, int);
__global__ void binarization(uint8_t*, uint8_t*, int, int);

__global__ void shared_dilation(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT);
__global__ void shared_erosion(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT);
__global__ void shared_combined(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT);

__global__ void rotate_gather(uint8_t*, uint8_t*, unsigned int, unsigned int, float, float);
__global__ void rotate_gather_90(uint8_t*, uint8_t*, unsigned int, unsigned int);
__global__ void rotate_scatter_90(uint8_t*, uint8_t*, unsigned int, unsigned int);
__global__ void rotate_shared_90(const uint8_t*, uint8_t*, const unsigned int, const unsigned int);
// __global__ void rescaling_bilin(uint8_t*, uint8_t*, unsigned int, unsigned int, float);
__global__ void rescaling_bilin(const uint8_t*, uint8_t*, const unsigned int, const unsigned int, const float, const unsigned int, const unsigned int);

__global__ void adaptive_threshold_histogram(unsigned int*, uint8_t*, unsigned int);
__global__ void adaptive_threshold_calculation(unsigned int*, unsigned int, uint8_t*);
__global__ void adaptive_threshold_apply(uint8_t*, uint8_t*, int, uint8_t*);

#endif
