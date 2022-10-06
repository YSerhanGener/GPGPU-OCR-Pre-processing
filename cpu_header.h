#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <iomanip>

#include <sys/time.h>

using namespace std;

// Update this as MASK_SIZE from kernel.cu is updated!
#define MASK_SIZE_CPU 3//7//3//5

#define imax(a,b) (a > b) ? a : b;
#define imin(a,b) (a < b) ? a : b;

void error_check(uint8_t*, uint8_t*, int, int, const char*);
void rgb2gray_inversion_cpu(uint8_t*, uint8_t*, int, int);
void noise_removal_cpu(uint8_t*, uint8_t*, const float*, int, int);

void binarization_cpu(uint8_t*, uint8_t*, int, int);
void adaptive_binarization_cpu(uint8_t*, uint8_t*, int, int, int);

void dilation_cpu(uint8_t*, const uint8_t*, uint8_t*, int, int, int);
void erosion_cpu(uint8_t*, const uint8_t*, uint8_t*, int, int, int);

void rotate_cpu(uint8_t*, uint8_t*, unsigned int, unsigned int, float, float);
void rescaling_bilin_cpu(uint8_t*, uint8_t*, unsigned int, unsigned int, float);