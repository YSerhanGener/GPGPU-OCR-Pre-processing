#include "cpu_header.h"

struct timeval begin_cpu, end_cpu;
long seconds;
long microseconds;
double elapsed;

void error_check(uint8_t *gpu_out, uint8_t *cpu_out, int width, int height, const char *name){
	int error=0;
	for(int i = 0; i < width*height; i++){
		if (abs(cpu_out[i]-gpu_out[i])>1){
			error++;
			/*if (error < 20){
				printf("%d: %d\t%d\n", i, cpu_out[i], gpu_out[i]);
			}*/
		}
	}

	if (error != 0){
		cout << error << " error(s) in " << name << " call!" << endl;
	}
	else{
		cout << name << " passed tests successfully!" << endl;
	}
}

void rgb2gray_inversion_cpu(uint8_t *cpu_out, uint8_t *input, int width, int height){
	for(int i = 0; i < width*height; i++){
		cpu_out[i] = 255 - (input[3*i]*0.21 + input[3*i+1]*0.71 + input[3*i+2]*0.07);
	}
}

void noise_removal_cpu(uint8_t *cpu_out, uint8_t *input, const float *gaus_kernel, int width, int height){
	float sum;
	int mask_offset = MASK_SIZE_CPU/2;
	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			sum = 0;
			for(int k = -mask_offset; k < MASK_SIZE_CPU-mask_offset; k++){
				for(int l = -mask_offset; l < MASK_SIZE_CPU-mask_offset; l++){
					if(i+k >= 0 && i+k < height && j+l >= 0 && j+l < width){
						sum += input[(i+k)*width+j+l] * gaus_kernel[(k+mask_offset)*MASK_SIZE_CPU + l + mask_offset];
					}
				}
			}
			cpu_out[i*width+j] = sum;
		}
	}
}

void binarization_cpu(uint8_t *cpu_out, uint8_t *input, int width, int height){
	for(int i = 0; i < width*height; i++){
		if(input[i] > 150){
			cpu_out[i] = 255;
		}
		else{
			cpu_out[i] = 0;
		}
	}
}

void adaptive_binarization_cpu(uint8_t *cpu_out, uint8_t *input, int width, int height, int t){
	for(int i = 0; i < width*height; i++){
		if(input[i] >= t){
			cpu_out[i] = 255;
		}
		else{
			cpu_out[i] = 0;
		}
	}
}

void erosion_cpu(uint8_t *out, const uint8_t *filter, uint8_t *src, int width, int height, int filter_width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			out[i*width + j]=0;	
			int start_col=j-(filter_width/2);
			int start_row = i-(filter_width/2);
			uint8_t minimum=255; //Keep to 1 for erosion. For dilation set max to 0 to avoid stagnation.
			for(int f=0; f < filter_width; f++){
				
				for (int g=0; g < filter_width; g++ ){
					
					int cur_row = start_row + f;
					int cur_col = start_col + g;
					if(filter[f*filter_width+g]==0){
						//You would want to ignore the zero element of the filer. 
						//temp[temp[f*filter_width+g]]=1; //For erosion //For dilation set to 0. 
							minimum=imin(minimum,255);
					}
					else{
						if(cur_row>-1 && cur_col>-1 &&cur_row<height &&cur_col<width){
							minimum=imin(minimum, src[cur_row*width+cur_col]);}
					}
				}
			}
			out[i*width + j] = minimum;
		}
	}
}

void dilation_cpu(uint8_t *out, const uint8_t *filter, uint8_t *src, int width, int height, int filter_width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			out[i*width + j]=0;	
			int start_col=j-(filter_width/2);
			int start_row = i-(filter_width/2);
			uint8_t maximum=0; //Keep to 1 for erosion. For dilation set max to 0 to avoid stagnation.
			for(int f=0; f < filter_width; f++){
				
				for (int g=0; g < filter_width; g++ ){
					
					int cur_row = start_row + f;
					int cur_col = start_col + g;
					if(filter[f*filter_width+g]==0){
						//You would want to ignore the zero element of the filer. 
						//temp[temp[f*filter_width+g]]=1; //For erosion //For dilation set to 0. 
						maximum=imax(maximum,0);
					}
					else{
						if(cur_row>-1 && cur_col>-1 &&cur_row<height &&cur_col<width){
						//maximum=imax(maximum, src[(i-f/2)*width + (j-g/2)]);
						maximum=imax(maximum, src[cur_row*width+cur_col]);}
					}
				}
			}
			out[i*width + j] = maximum; 
		}
	}
}

void rotate_cpu(uint8_t *dst, uint8_t *src, unsigned int width, unsigned int height, float sin_theta, float cos_theta) {
	// Pixel coordinates in rotated image
	unsigned int x_dst;
	unsigned int y_dst;

	// Center of image
	const float x_center = float(width) / 2.0;
	const float y_center = float(height) / 2.0;

	for (y_dst = 0; y_dst < height; y_dst++) {
		for (x_dst = 0; x_dst < width; x_dst++) {

			// Offset from center
			float x_off = x_dst - x_center;
			float y_off = y_dst - y_center;

			// Pixel coordinates in original image
			unsigned int x_src = (unsigned int)((x_off * cos_theta) + (y_off * sin_theta) + x_center);
			unsigned int y_src = (unsigned int)((y_off * cos_theta) - (x_off * sin_theta) + y_center);

			// Check boundaries
			if (((unsigned int)x_src >= 0) && ((unsigned int)x_src < width)
					&& ((unsigned int)y_src >= 0) && ((unsigned int)y_src < height)) {
				// Copy pixel
				dst[y_dst * width + x_dst] = src[y_src * width + x_src];
			} else {
				dst[y_dst * width + x_dst] = 0;
			}
		}
	}
}

void rescaling_bilin_cpu(uint8_t *dst, uint8_t *src, unsigned int width, unsigned int height, float scale) {
	// Pixel coordinates in rescaled image
	unsigned int x_dst;
	unsigned int y_dst;

	// Width and height in new image
	const unsigned int new_width = round(width * scale);
	const unsigned int new_height = (height * scale);

	// Center of image
	const float x_center = float(width) / 2.0;
	const float y_center = float(height) / 2.0;

	for (y_dst = 0; y_dst < height; y_dst++) {
		for (x_dst = 0; x_dst < width; x_dst++) {
			// Offset from center
			float x_dst_off = x_dst - x_center;
			float y_dst_off = y_dst - y_center;

			// Pixel offset in original image
			float x_src_off = float(x_dst_off) / scale;
			float y_src_off = float(y_dst_off) / scale;

			// Pixel coordinates in original image
			float x_src = x_src_off + x_center;
			float y_src = y_src_off + y_center;

			// Linear scaling factor
			float rght_frac = 1 - (ceil(x_src) - x_src);
			float left_frac = 1 - rght_frac;
			float top_frac = 1 - (y_src - floor(y_src));
			float bot_frac = 1 - top_frac;

			if ((((unsigned int)abs(x_src_off)) < width / 2.0)
					&& (((unsigned int)abs(y_src_off)) < height / 2.0)
					&& ((unsigned int)abs(x_dst_off) < new_width / 2.0)
					&& ((unsigned int)abs(y_dst_off) < new_height / 2.0)) {

				float top_bilin_intrpl;
				top_bilin_intrpl = src[(unsigned int)(floor(y_src) * width + ceil(x_src))] * rght_frac;
				top_bilin_intrpl += src[(unsigned int)(floor(y_src) * width + floor(x_src))] * left_frac;

				float bot_bilin_intrpl;
				bot_bilin_intrpl = src[(unsigned int)(ceil(y_src) * width + floor(x_src))] * rght_frac;
				bot_bilin_intrpl += src[(unsigned int)(ceil(y_src) * width + ceil(x_src))] * left_frac;

				uint8_t bilin_intrpl;
				bilin_intrpl = (uint8_t)((float)(top_bilin_intrpl * top_frac) + (bot_bilin_intrpl * bot_frac));

				dst[y_dst * width + x_dst] = bilin_intrpl;
			} else {
				dst[y_dst * width + x_dst] = 0;
			}
		}
	}
}