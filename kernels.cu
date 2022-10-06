#include "kernels.h"

__global__ void rgb2gray_invesion(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	int COL = threadIdx.x + blockIdx.x * blockDim.x; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * blockDim.y; // Row of the thread
	int indexO = ROW * WIDTH + COL; // index for the output array
	int indexI = (ROW * WIDTH + COL)*3; // index for the input array since array is structured in rgbrgbrgb... manner each thread operates on 3 consecutive elements on the same array
	if (ROW<HEIGHT && COL<WIDTH){ // Boundary checks
		// 255 - (grayscale value) color inversion
		output[indexO] = 255 - (input[indexI]*0.21 // Red value of the input array
					+ input[indexI+1]*0.71 // Green value of the input array
					+ input[indexI+2]*0.07); // Blue value of the input array
	}
}

__global__ void noise_removal(uint8_t *input, uint8_t *output, float *gaus_kernel, int WIDTH, int HEIGHT){
	int COL = threadIdx.x + blockIdx.x * blockDim.x; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * blockDim.y; // Row of the thread
	int index = ROW * WIDTH + COL; // index for the pixel to apply kernel
	float sum = 0;
	int mask_offset = MASK_SIZE/2;
	if (ROW<HEIGHT && COL<WIDTH){ // Boundary checks
		sum = 0;
		for(int i = -mask_offset; i < MASK_SIZE-mask_offset; i++){
			for(int j = -mask_offset; j < MASK_SIZE-mask_offset; j++){
				if(ROW+i >= 0 && ROW+i < HEIGHT && COL+j >= 0 && COL+j < WIDTH){
					sum += input[(ROW+i)*WIDTH+COL+j] * gaus_kernel[(i+mask_offset)*MASK_SIZE + j+mask_offset]; // Multiply each pixel with the corresponding weigth in the kernel
				}
			}
		}
		output[index] = sum;
	}
}

__global__ void constant_memory_noise_removal(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	int COL = threadIdx.x + blockIdx.x * blockDim.x; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * blockDim.y; // Row of the thread
	int index = ROW * WIDTH + COL; // index for the pixel to apply kernel
	float sum = 0;
	int mask_offset = MASK_SIZE/2;
	if (ROW<HEIGHT && COL<WIDTH){ // Boundary checks
		sum = 0;
		for(int i = -mask_offset; i < MASK_SIZE-mask_offset; i++){
			for(int j = -mask_offset; j < MASK_SIZE-mask_offset; j++){
				if(ROW+i >= 0 && ROW+i < HEIGHT && COL+j >= 0 && COL+j < WIDTH){
					sum += input[(ROW+i)*WIDTH+COL+j] * mask_nr[(i+mask_offset)*MASK_SIZE + j+mask_offset]; // Multiply each pixel with the corresponding weigth in the kernel
				}
			}
		}
		output[index] = sum;
	}
}

__global__ void tiled_noise_removal(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	float result = 0;
	int COL_out = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW_out = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread
	
	int COL = COL_out - MASK_SIZE / 2;
	int ROW = ROW_out - MASK_SIZE / 2;

	// Shared memory for the current tile
	__shared__ uint8_t TILE[BLOCK_SIZE][BLOCK_SIZE];
	
	// 0 pad the shared memory based on the tile coordinates
	if(ROW >= 0 && ROW < HEIGHT && COL >= 0 && COL < WIDTH){
		TILE[threadIdx.y][threadIdx.x] = input[ROW * WIDTH + COL];
	}
	else{
		TILE[threadIdx.y][threadIdx.x] = 0;
	}
	// Wait for all threads to complete their initilization of the shared memory space
	__syncthreads();

	// Boundary conditions for threads that takes part in the execution part
	if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE){
		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
				result += mask_nr[i * MASK_SIZE + j] * TILE[threadIdx.y + i][threadIdx.x + j];
			}
		}
		if(ROW_out < HEIGHT && COL_out < WIDTH){
			output[ROW_out * WIDTH + COL_out] = floorf(result);
		}
	}
}

__global__ void naive_binarization(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	int COL = threadIdx.x + blockIdx.x * blockDim.x; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * blockDim.y; // Row of the thread
	int index = ROW * WIDTH + COL; // index for the output array
	if (ROW<HEIGHT && COL<WIDTH){ // Boundary checks
		if(input[index]>150){
			output[index]=255;
		}
		else{
			output[index]=0;
		}
	}
}

__global__ void binarization(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	int COL = threadIdx.x + blockIdx.x * blockDim.x; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * blockDim.y; // Row of the thread
	int index = ROW * WIDTH + COL; // index for the output array
	if (ROW<HEIGHT && COL<WIDTH){ // Boundary checks
		output[index] = __vcmpgeu2(input[index], 151);
	}
}

__global__ void tiled_rgb2gray_inversion_noise_removal(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	float result = 0;
	int COL_out = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW_out = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread
	
	int COL = COL_out - MASK_SIZE / 2;
	int ROW = ROW_out - MASK_SIZE / 2;

	// Shared memory for the current tile
	__shared__ uint8_t TILE[BLOCK_SIZE][BLOCK_SIZE];

	int indexI = (ROW * WIDTH + COL)*3;
	// ) pad the shared memory based on the tile coordinates
	if(ROW >= 0 && ROW < HEIGHT && COL >= 0 && COL < WIDTH){
		TILE[threadIdx.y][threadIdx.x] = 255 - (input[indexI]*0.21 // Red value of the input array
					+ input[indexI+1]*0.71 // Green value of the input array
					+ input[indexI+2]*0.07);//input[ROW * WIDTH + COL];
	}
	else{
		TILE[threadIdx.y][threadIdx.x] = 0;
	}
	// Wait for all threads to complete their initilization of the shared memory space
	__syncthreads();

	// Boundary conditions for threads that takes part in the execution part
	if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE/* && ROW < HEIGHT && COL < WIDTH*/){
		// Noise Removal
		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
				result += mask_nr[i * MASK_SIZE + j] * TILE[threadIdx.y + i][threadIdx.x + j];
			}
		}
		// Final output write back and Binarization
		if(ROW_out < HEIGHT && COL_out < WIDTH){
			int currPixel=0;
			currPixel=floorf(result);
			output[ROW_out * WIDTH + COL_out] = currPixel;
		}
	}
}

__global__ void tiled_rgb2gray_inversion_noise_removal_histogram(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT, unsigned int* bins, unsigned int numPixels){
	float result = 0;
	int COL_out = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW_out = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread
	
	int COL = COL_out - MASK_SIZE / 2;
	int ROW = ROW_out - MASK_SIZE / 2;

	// Shared memory for the current tile
	__shared__ uint8_t TILE[BLOCK_SIZE][BLOCK_SIZE];

	__shared__ unsigned int subHist[NUM_BINS];
	
	int tid=threadIdx.x * TILE_SIZE + threadIdx.y;
	int gid=blockIdx.x * TILE_SIZE + blockIdx.y;
	if (tid < NUM_BINS){
		subHist[tid]=0;
	}
			
	int indexI = (ROW * WIDTH + COL)*3;
	// ) pad the shared memory based on the tile coordinates
	if(ROW >= 0 && ROW < HEIGHT && COL >= 0 && COL < WIDTH){
		TILE[threadIdx.y][threadIdx.x] = 255 - (input[indexI]*0.21 // Red value of the input array
					+ input[indexI+1]*0.71 // Green value of the input array
					+ input[indexI+2]*0.07);//input[ROW * WIDTH + COL];
	}
	else{
		TILE[threadIdx.y][threadIdx.x] = 0;
	}
	// Wait for all threads to complete their initilization of the shared memory space
	__syncthreads();

	// Boundary conditions for threads that takes part in the execution part
	if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE/* && ROW < HEIGHT && COL < WIDTH*/){
		// Noise Removal
		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
				result += mask_nr[i * MASK_SIZE + j] * TILE[threadIdx.y + i][threadIdx.x + j];
			}
		}
		// Final output write back and Binarization
		if(ROW_out < HEIGHT && COL_out < WIDTH){
			int currPixel=0;
			currPixel=floorf(result);
			output[ROW_out * WIDTH + COL_out] = currPixel;
			if (gid < numPixels){
				atomicAdd(&subHist[currPixel],1);
			}
			__syncthreads();
		
			//no need to write histogram back to global memory, use subhistogram to create sub zero and first orders, then accumulate
			if(tid < NUM_BINS){
				atomicAdd(&bins[tid],subHist[tid]);
			}
		}
	}
}

__global__ void shared_erosion(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
		
	uint8_t minimum=255;
	int COL = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread

	int start_col=COL-(MASK_SIZE/2);
	int start_row = ROW-(MASK_SIZE/2);
	__shared__ uint8_t Ns[BLOCK_SIZE][BLOCK_SIZE];

	if(start_row >= 0 && start_row < HEIGHT && start_col >= 0 && start_col < WIDTH){
		Ns[threadIdx.y][threadIdx.x]= input[start_row * WIDTH + start_col];	
	}
	else{
		Ns[threadIdx.y][threadIdx.x]=255;
	}

	__syncthreads();
	if(threadIdx.y < TILE_SIZE&& threadIdx.x < TILE_SIZE){

		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
					minimum=imin(minimum, Ns[threadIdx.y+i][threadIdx.x+j]);
			}
		}
		if(ROW < HEIGHT && COL < WIDTH){
			output[ROW * WIDTH + COL] = minimum;
		}
	}
	
}

__global__ void shared_dilation(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	
	
	uint8_t maximum=0;
	int COL = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread

	int start_col=COL-(MASK_SIZE/2);
	int start_row = ROW-(MASK_SIZE/2);
	__shared__ uint8_t Ns[BLOCK_SIZE][BLOCK_SIZE];
	if(start_row >= 0 && start_row < HEIGHT && start_col >= 0 && start_col < WIDTH){
		Ns[threadIdx.y][threadIdx.x]= input[start_row * WIDTH + start_col];	
	}
	else{
		Ns[threadIdx.y][threadIdx.x]=0;
	}

	__syncthreads();
	if(threadIdx.y < TILE_SIZE&& threadIdx.x < TILE_SIZE){

		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
					maximum=imax(maximum, Ns[threadIdx.y+i][threadIdx.x+j]);
			}
		}
		//__syncthreads();
		if(ROW < HEIGHT && COL < WIDTH){
			output[ROW * WIDTH + COL] = maximum;
		}
	}
}

__global__ void shared_opening(uint8_t *input, uint8_t *output, int WIDTH, int HEIGHT){
	
	uint8_t maximum=0;
	uint8_t minimum =255;
	int COL = threadIdx.x + blockIdx.x * TILE_SIZE; // Col of the thread
	int ROW = threadIdx.y + blockIdx.y * TILE_SIZE; // Row of the thread

	int NEW_COL = threadIdx.x + blockIdx.x * LOW_TILE_SIZE; // Col of the thread
	int NEW_ROW = threadIdx.y + blockIdx.y * LOW_TILE_SIZE; // Row of the thread

	int start_col=COL-(MASK_SIZE/2);
	int start_row = ROW-(MASK_SIZE/2);
	__shared__ uint8_t Ns[BLOCK_SIZE][BLOCK_SIZE];
	if(start_row >= 0 && start_row < HEIGHT && start_col >= 0 && start_col < WIDTH){
		Ns[threadIdx.y][threadIdx.x]= input[start_row * WIDTH + start_col];	
	}
	else{
		Ns[threadIdx.y][threadIdx.x]=255;
	}

	__syncthreads();

	if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE){

		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
				if(mask_de[i*MASK_SIZE+j]==0){
					minimum=imin(minimum,255);
					}
				else{
					minimum=imin(minimum, Ns[threadIdx.y+i][threadIdx.x+j]);				
				}
			}
		}
		Ns[threadIdx.y][threadIdx.x]=minimum;

	}
	if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE && threadIdx.y > LOW_TILE_SIZE && threadIdx.x < LOW_TILE_SIZE){
		Ns[threadIdx.y][threadIdx.x]=0;
	}
	
	__syncthreads();

	if(threadIdx.y < LOW_TILE_SIZE && threadIdx.x < LOW_TILE_SIZE){

		for(int i = 0; i < MASK_SIZE; i++){
			for(int j = 0; j < MASK_SIZE; ++j){
				if(mask_de[i*MASK_SIZE+j]==0){
					maximum=imax(maximum,0);
					}
				else{
					maximum=imax(maximum, Ns[threadIdx.y+i][threadIdx.x+j]);				
				}
			}
		}
		if(NEW_ROW < HEIGHT && NEW_COL < WIDTH){
			output[NEW_ROW * WIDTH + NEW_COL] = maximum;
		}
	}
}

__global__ void rotate_gather(uint8_t *src, uint8_t *dst, unsigned int width, unsigned int height, float sin_theta, float cos_theta) {
	// Pixel coordinates in rotated image
	const unsigned int x_dst = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y_dst = blockDim.y * blockIdx.y + threadIdx.y;

	// Center of image
	const float x_center = float(width) / 2.0;
	const float y_center = float(height) / 2.0;

	// Offset from center
	float x_dst_off = x_dst - x_center;
	float y_dst_off = y_dst - y_center;

	// Pixel coordinates in original image
	unsigned int x_src = (unsigned int)((x_dst_off * cos_theta) + (y_dst_off * sin_theta) + x_center);
	unsigned int y_src = (unsigned int)((y_dst_off * cos_theta) - (x_dst_off * sin_theta) + y_center);

	float x_src_off = x_src - x_center;
	float y_src_off = y_src - y_center;

	if ((((unsigned int)abs(x_src_off)) < width / 2.0)
			&& (((unsigned int)abs(y_src_off)) < height / 2.0)
			&& ((unsigned int)abs(x_dst_off) < width / 2.0)
			&& ((unsigned int)abs(y_dst_off) < height / 2.0)) {
		// Copy pixel
		dst[y_dst * width + x_dst] = src[y_src * width + x_src];
	} else {
		dst[y_dst * width + x_dst] = 0;
	}
}

__global__ void rotate_gather_90(uint8_t *src, uint8_t *dst, unsigned int new_width, unsigned int new_height) {
	// Pixel coordinates in rotated image
	const unsigned int x_dst = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y_dst = blockDim.y * blockIdx.y + threadIdx.y;

	const unsigned int width = new_height;
	const unsigned int height = new_width;

	// Pixel coordinates in original image
	unsigned int x_src = width - y_dst -1;
	unsigned int y_src = x_dst;

	if(x_dst<new_width && y_dst<new_height){
		// Copy pixel
		dst[y_dst * new_width + x_dst] = src[y_src * width + x_src];
	}
}

__global__ void rotate_scatter_90(uint8_t *src, uint8_t *dst, unsigned int new_width, unsigned int new_height) {
	// Pixel coordinates in rotated image
	const unsigned int x_src = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y_src = blockDim.y * blockIdx.y + threadIdx.y;

	const unsigned int width = new_height;
	const unsigned int height = new_width;

	// Pixel coordinates in original image
	const unsigned int x_dst = y_src;
	const unsigned int y_dst = width - x_src - 1;

	if (x_dst < new_width && y_dst < new_height) {
		// Copy pixel
		dst[y_dst * new_width + x_dst] = src[y_src * width + x_src];
	}
}

__global__ void rotate_shared_90(const uint8_t *src, uint8_t *dst, const unsigned int new_width, const unsigned int new_height) {
	// Pixel coordinates in rotated image
  __shared__ unsigned int blk[BLOCK_SIZE][BLOCK_SIZE];

	const unsigned int x_dst = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y_dst = blockDim.y * blockIdx.y + threadIdx.y;

  const unsigned int x_src_crn = new_height - (blockDim.y * (blockIdx.y + 1));
  const unsigned int y_src_crn = blockDim.x * blockIdx.x;

	const unsigned int x_rd = x_src_crn + threadIdx.x;
	const unsigned int y_rd = y_src_crn + threadIdx.y;

	const unsigned int width = new_height;
	const unsigned int height = new_width;

  if (y_rd < new_width) 
    blk[threadIdx.y][threadIdx.x] = src[y_rd * width + x_rd];

  __syncthreads();

	// Pixel coordinates in original image
	// unsigned int x_src = width - y_dst - 1;
	// unsigned int y_src = x_dst;
	unsigned int x_src = blockDim.y - threadIdx.y - 1;
	unsigned int y_src = threadIdx.x;

	if (x_dst < new_width && y_dst < new_height) {
		// Copy pixel
		dst[y_dst * new_width + x_dst] = blk[y_src][x_src];
	}
}

__global__ void rescaling_bilin(const uint8_t *src, uint8_t *dst, const unsigned int src_width, const unsigned int src_height, const float scale, const unsigned int dst_width, const unsigned int dst_height) {
	// Pixel coordinates in rescaled image
	const unsigned int x_dst = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int y_dst = blockDim.y * blockIdx.y + threadIdx.y;

	// Width and height in new image
	const unsigned int new_width  = roundf(src_width * scale);
	const unsigned int new_height = roundf(src_height * scale);

	// Center of image
	const float x_src_center = float(src_width) / 2.0;
	const float y_src_center = float(src_height) / 2.0;
	const float x_dst_center = float(dst_width) / 2.0;
	const float y_dst_center = float(dst_height) / 2.0;

	// Offset from center
	float x_dst_off = x_dst - x_dst_center;
	float y_dst_off = y_dst - y_dst_center;

	// Pixel offset in original image
	float x_src_off = x_dst_off / scale;
	float y_src_off = y_dst_off / scale;

	// Pixel coordinates in original image
	float x_src = x_src_off + x_src_center;
	float y_src = y_src_off + y_src_center;

	// Linear scaling factor
	float rght_frac = 1 - (ceil(x_src) - x_src);
	float left_frac = 1 - rght_frac;
	float top_frac = 1 - (y_src - floor(y_src));
	float bot_frac = 1 - top_frac;

	if ((abs(x_src_off) < (src_width / 2.0))
			&& (abs(y_src_off) < (src_height / 2.0))
			&& (abs(x_dst_off) < (dst_width / 2.0))
			&& (abs(y_dst_off) < (dst_height / 2.0))
			&& (abs(x_dst_off) < (new_width / 2.0))
			&& (abs(y_dst_off) < (new_height / 2.0))) {

		float top_bilin_intrpl;
		top_bilin_intrpl = src[(unsigned int)(floor(y_src) * src_width + ceil(x_src))] * rght_frac;
		top_bilin_intrpl += src[(unsigned int)(floor(y_src) * src_width + floor(x_src))] * left_frac;

		float bot_bilin_intrpl;
		bot_bilin_intrpl = src[(unsigned int)(ceil(y_src) * src_width + floor(x_src))] * rght_frac;
		bot_bilin_intrpl += src[(unsigned int)(ceil(y_src) * src_width + ceil(x_src))] * left_frac;

		uint8_t bilin_intrpl;
		bilin_intrpl = (uint8_t)((float)(top_bilin_intrpl * top_frac) + (bot_bilin_intrpl * bot_frac));

		dst[y_dst * dst_width + x_dst] = bilin_intrpl;
	} else if ((x_dst < dst_width) && (y_dst < dst_height)) {
		dst[y_dst * dst_width + x_dst] = 0;
	}
}

__global__ void adaptive_threshold_histogram(unsigned int* bins, uint8_t *inImage, unsigned int numPixels){
  /*
  Full implementation of Otsu adaptive thresholding integrating all other kernel optimizations. The kernel should be called with enough
  threads such that there is a thread for each pixel. between 256 and 1024 threads should be launched per block (need to run new sweep)
  start with shared histogram implementation. Image must be unsigned int or other compatible type with atomic operations
  */
    //may be too much shared memory for a single SM with 32 blocks, maybe it works?.
  // __shared__ unsigned int image[BLOCK_SIZE];
  __shared__ unsigned int subHist[NUM_BINS];
  // __shared__ float values[NUM_BINS];
  // __shared__ float zeroOrder[NUM_BINS];
  // __shared__ float firstOrder[NUM_BINS];


  int tid=threadIdx.x;
  int gid=threadIdx.x + blockIdx.x*blockDim.x;
  if (tid < NUM_BINS){
    subHist[tid]=0;
  }
  //each block works on a small part of the image
  __syncthreads();
  int currPixel=0;
  if (gid < numPixels){
    // image[tid] = inImage[gid];
    currPixel=inImage[gid];
    atomicAdd(&subHist[currPixel],1);
  }
    __syncthreads();
    
    //no need to write histogram back to global memory, use subhistogram to create sub zero and first orders, then accumulate
  if(tid < NUM_BINS){
    atomicAdd(&bins[tid],subHist[tid]);
  }
}

__global__ void adaptive_threshold_calculation(unsigned int* bins, unsigned int numPixels, uint8_t* threshold){

  int tid=threadIdx.x;

   __shared__ float zeroOrder[NUM_BINS];
   __shared__ float values[NUM_BINS];
   __shared__ float firstOrder[NUM_BINS];
   __shared__ unsigned int index[NUM_BINS];

  float zeroVal = (float)bins[tid]/(float)numPixels;
  zeroOrder[tid] = zeroVal;
  firstOrder[tid] = zeroVal*tid;
  

  __syncthreads();
  

  //similarly, do reduce and construct with local copy of each
  for(int stride = 1; stride<=NUM_BINS/2; stride<<=1){
      int index = (tid+1)*stride*2-1;
        if(index < NUM_BINS){
            zeroOrder[index] += zeroOrder[index -stride];
            firstOrder[index] += firstOrder[index -stride];
        }
        __syncthreads();
    }

    for(int stride=NUM_BINS/4; stride > 0; stride>>=1){
      __syncthreads();
      int index = (tid+1)*stride*2-1;
      if(index+stride < NUM_BINS){
        zeroOrder[index+stride]+=zeroOrder[index];
        firstOrder[index+stride]+=firstOrder[index];
      }
    }
  
    __syncthreads();


     //now read totals back into local for inter class. Only 1 block needs to do thi
      
         index[tid]=tid;
         values[tid]=0;
        __syncthreads();
  
        float numerator = pow(firstOrder[NUM_BINS-1]*zeroOrder[tid]-firstOrder[tid],2);
        float denominator = zeroOrder[tid]*(1.0-zeroOrder[tid]);
        if(denominator==0){
            values[tid]=0;
        }
        else{
            values[tid]=numerator/denominator;
        }
        __syncthreads();
  

        for(int stride = NUM_BINS / 2; stride > 0; stride>>=1){
            if(tid < stride){
                if(values[tid+stride] > values[tid]){
                    index[tid] = index[tid+stride];
                    values[tid] = values[tid+stride];
                }


            }
            __syncthreads();
        }
        if(tid==0){
            *threshold=index[0];
        }
     }

__global__ void adaptive_threshold_apply(uint8_t *inImage, uint8_t *outImage, int numPixels, uint8_t* threshold) {
  int i = threadIdx.x + blockDim.x * blockIdx.x; //1 to 1 indexing
  // int stride = gridDim.x*blockDim.x;  //will not matter when there are exactly enough threads to read the image

  if (i < numPixels){
		//outImage[i] = __vcmpleu2(inImage[i], *threshold);
		outImage[i] = __vcmpgeu2(inImage[i], *threshold);
    }
}
