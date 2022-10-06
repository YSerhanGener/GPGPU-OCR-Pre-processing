#ifdef RUN_TESSERACT
#include <tesseract/baseapi.h>
#endif

#include "kernels.cu"

#ifdef RUN_TESTS
#include "cpu_header.h"
#endif


int main(int argc, char *argv[]) {
#ifdef RUN_TESSERACT
	char *outText;
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
	if (api->Init(NULL, "eng")) {
       exit(1);
    }
#endif

	int width, height, components, rgb=3;
	char imagepath[100] = "image.png";
	if (argc == 2){
		strcpy(imagepath, argv[1]);
	}
	else if (argc > 2){
		cout << "Usage: " << argv[0] << " [image file]" << endl;
		return -1;
	}
	cout << "Using image at " << imagepath << endl;

#if defined(TIME_CUDA) || defined(RUN_TESTS)
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	float total_time = 0;
#endif
	/* Read image from using stbi
	 * Image Name: Path to image file
	 * Width: Width of the image
	 * Height: Height of the image
	 * Components: Components of the input image
	 *     N=#comp     components	
	 *		1           grey	
	 *		2           grey, alpha	
	 *		3           red, green, blue	
	 *		4           red, green, blue, alpha
	 * RGB: Requested components
	 */
	uint8_t* rgb_image = stbi_load(imagepath, &width, &height, &components, rgb);
	if (rgb_image == NULL){
		cout << "Failed to open image " << imagepath << endl;
		return -1;
	}
	cout << "Image Width:" << width << "\tHeight:" << height << "\t#ofChannels:" << components << std::endl;
	int numPixels = width*height;

#ifdef RUN_TESTS
	uint8_t *cpu_out;
	cpu_out = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	uint8_t *cpu_out2;
	cpu_out2 = (uint8_t *) malloc(width*height*sizeof(uint8_t));
#endif

	const float rescale = 2.5;
	/* decleration of variables */
	uint8_t *device_rgb, *device_gs, *device_nr, *device_bin, *device_gt, *device_dil, *device_ero, *device_rs, *device_rt;
	uint8_t *host_inversion, *host_nr, *host_bin, *host_dil, *host_ero, *host_rs, *host_rt;
	uint8_t *host_combined;
	unsigned int *device_hist;
	uint8_t *device_threshold, *host_threshold;
	float *device_nr_mask;
	uint8_t *device_de_mask;

#if MASK_SIZE == 3
	const float host_nr_mask[MASK_DIM] = {1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM};
	const uint8_t host_de_mask[MASK_DIM] = {1,1,1,1,1,1,1,1,1};
#endif
#if MASK_SIZE == 5
	const float host_nr_mask[MASK_DIM] = {1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM};
	const uint8_t host_de_mask[MASK_DIM] = {1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1};
#endif
#if MASK_SIZE == 7
	const float host_nr_mask[MASK_DIM] = {1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM};
	const uint8_t host_de_mask[MASK_DIM] = {1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1 , 1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1};
#endif

	host_inversion = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_nr = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_bin = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_combined = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_dil = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_ero = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_rs = (uint8_t *) malloc(roundf(width * rescale) * roundf(height * rescale)*sizeof(uint8_t));
	host_rt = (uint8_t *) malloc(width*height*sizeof(uint8_t));
	host_threshold = (uint8_t *) malloc(sizeof(uint8_t));
	cudaError_t err;

	// Setup constant memory for noise removal
	err = cudaMemcpyToSymbol(mask_nr, &host_nr_mask, MASK_DIM * sizeof(float));
	if(err!=cudaSuccess){
		cout << "Failed to setup consttant memory for noise removal " << __LINE__ << endl;
		return -1;
	}
	// Setup constant memory for opening
	err = cudaMemcpyToSymbol(mask_de, &host_de_mask, MASK_DIM * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to setup consttant memory for opening " << __LINE__ << endl;
		return -1;
	}

	/* Cuda malloc for input and rgb2grayscale+inversion output */
	err = cudaMalloc((void **)&device_rgb, width * height * rgb * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for input image at line " << __LINE__ << endl;
		return -1;
	}
	err = cudaMalloc((void **)&device_gs, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for gs image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		return -1;
	}
	/* Cuda malloc for noise removal output and mask*/
	err = cudaMalloc((void **)&device_nr, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for nr image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_bin, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for binarization image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_nr_mask, MASK_DIM * sizeof(float));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for nr mask at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_de_mask, MASK_DIM * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for de mask at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr mask from device memory before failing
		return -1;
	}
	/* Cuda malloc for global thresholding*/
	err = cudaMalloc((void **)&device_gt, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for gt image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		return -1;
	}
	/* Cuda malloc for dilation*/
	err = cudaMalloc((void **)&device_dil, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for dil image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		return -1;
	}
	/* Cuda malloc for erosion*/
	err = cudaMalloc((void **)&device_ero, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for ero image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		return -1;
	}
	/* Cuda malloc for rescale*/
	err = cudaMalloc((void **)&device_rs, roundf(width * rescale) * roundf(height * rescale) * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for rs image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		return -1;
	}
	/* Cuda malloc for rotate*/
	err = cudaMalloc((void **)&device_rt, width * height * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for rt image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		return -1;
	}
	/* Cuda malloc for adaptive thresholding*/
	err = cudaMalloc((void **)&device_hist, width * height * sizeof(unsigned int));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for histogram at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_threshold, sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for threshold at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		return -1;
	}

	/* Cuda mem copy for gaus mask and de mask*/
	err = cudaMemcpy(device_nr_mask, host_nr_mask, MASK_DIM*sizeof(float), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cout << "Failed to copy nr mask to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}
	err = cudaMemcpy(device_de_mask, host_de_mask, MASK_DIM*sizeof(uint8_t), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cout << "Failed to copy de mask to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

// RGB2GRAY and Inversion Kernels

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	/* Cuda mem copy for rgb input image */
	err = cudaMemcpy(device_rgb, rgb_image, width*height*rgb*sizeof(uint8_t), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cout << "Failed to copy input image to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of Host2Device memcpy: %.6f seconds\n", milliseconds*1e-3);
#endif

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	// Set block and grid dim for rgb to gray+inversion kernel
	int block_size = 32;
	dim3 grid_dim(((width-1)/block_size)+1,((height-1)/block_size)+1,1);
	dim3 block_dim(block_size,block_size,1);

	// Call rgb to gray+inversion kernel
	rgb2gray_invesion<<<grid_dim, block_dim>>>(device_rgb, device_gs, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of GPU rgb2gray + inversion: %.6f seconds\n", milliseconds*1e-3);
#endif

	// Get intermediate data to test rgb2gray_inversion kernel
	err = cudaMemcpy(host_inversion, device_gs, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "Failed to copy inversion image from device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaEventRecord(start);
	rgb2gray_inversion_cpu(cpu_out, rgb_image, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of rgb2gray + inversion cpu: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_inversion, cpu_out, width, height, "rgb2gray + inversion");
#endif

// Naive Noise Removal

#ifdef TIME_CUDA
	cudaEventRecord(start);
	// Use same block and grid dim for noise removal kernel
	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/BLOCK_SIZE)+1;
	grid_dim.y = ((height-1)/BLOCK_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;
	noise_removal<<<grid_dim, block_dim>>>(device_gs, device_nr, device_nr_mask, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of GPU noise removal: %.6f seconds\n", milliseconds*1e-3);
#endif

// Constant memory Noise Removal

#ifdef TIME_CUDA
	cudaEventRecord(start);
	// Use same block and grid dim for noise removal kernel
	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/BLOCK_SIZE)+1;
	grid_dim.y = ((height-1)/BLOCK_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;
	constant_memory_noise_removal<<<grid_dim, block_dim>>>(device_gs, device_nr, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of GPU constant memory noise removal: %.6f seconds\n", milliseconds*1e-3);
#endif

// Tiled Noise Removal
	
#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif
	// Use same block and grid dim for noise removal kernel
	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/TILE_SIZE)+1;
	grid_dim.y = ((height-1)/TILE_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;
	tiled_noise_removal<<<grid_dim, block_dim>>>(device_gs, device_nr, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of GPU tiled noise removal: %.6f seconds\n", milliseconds*1e-3);
#endif

	// Get intermediate data to test noise removal kernel
	err = cudaMemcpy(host_nr, device_nr, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "Failed to copy nr image from device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaEventRecord(start);
	noise_removal_cpu(cpu_out, host_inversion, host_nr_mask, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of CPU noise removal: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_nr, cpu_out, width, height, "noise removal");
#endif

// Naive Binarization

#ifdef TIME_CUDA
	cudaEventRecord(start);

	// Set block and grid dim for binarization kernel
	block_size = 32;
	grid_dim.x = ((width-1)/block_size)+1;
	grid_dim.y = ((height-1)/block_size)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;

	// Call binarization kernel
	naive_binarization<<<grid_dim, block_dim>>>(device_nr, device_bin, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of GPU Naive Binarization: %.6f seconds\n", milliseconds*1e-3);
#endif

// Binarization

#ifdef TIME_CUDA
	cudaEventRecord(start);

	// Set block and grid dim for binarization kernel
	block_size = 32;
	grid_dim.x = ((width-1)/block_size)+1;
	grid_dim.y = ((height-1)/block_size)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;

	// Call binarization kernel
	binarization<<<grid_dim, block_dim>>>(device_nr, device_bin, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of GPU binarization: %.6f seconds\n", milliseconds*1e-3);
#endif

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	// Set block and grid dim for adaptive binarization kernel
	block_size = 1024;
	grid_dim.x = (numPixels-1)/block_size +1;
	grid_dim.y = 1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = 1;
	block_dim.z = 1;

	adaptive_threshold_histogram<<<grid_dim,block_dim>>>(device_hist, device_nr, numPixels);
	adaptive_threshold_calculation<<<1,NUM_BINS>>>(device_hist, numPixels, device_threshold);
	adaptive_threshold_apply<<<grid_dim,block_dim>>>(device_nr, device_bin, numPixels, device_threshold);

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of GPU adaptive threshold: %.6f seconds\n", milliseconds*1e-3);
#endif

	// Get intermediate data to test thresholding kernel
	err = cudaMemcpy(host_bin, device_bin, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "Failed to copy binarization image from device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaMemcpy(host_threshold, device_threshold, sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cudaEventRecord(start);
	adaptive_binarization_cpu(cpu_out, host_nr, width, height, *host_threshold);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of CPU binarization: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_bin, cpu_out, width, height, "binarization");
#endif

// Opening - Dilation and Erosion

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/TILE_SIZE)+1;
	grid_dim.y = ((height-1)/TILE_SIZE)+1;
	//grid_dim.x = ((width-1)/LOW_TILE_SIZE)+1;
	//grid_dim.y = ((height-1)/LOW_TILE_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;
	
	shared_erosion<<<grid_dim, block_dim>>>(device_bin, device_ero, width, height);
	shared_dilation<<<grid_dim, block_dim>>>(device_ero, device_dil, width, height);
	//shared_combined<<<grid_dim, block_dim>>>(device_bin, device_dil, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of GPU opening: %.6f seconds\n", milliseconds*1e-3);
#endif

	// Get intermediate data to test binarization kernel
	err = cudaMemcpy(host_dil, device_dil, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "Failed to copy dil image from device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaEventRecord(start);
	//erosion_cpu(cpu_out, host_de_mask, host_bin, width, height, MASK_SIZE);
	//dilation_cpu(cpu_out, host_de_mask, host_bin, width, height, MASK_SIZE);
	erosion_cpu(cpu_out2, host_de_mask, host_bin, width, height, MASK_SIZE);
	dilation_cpu(cpu_out, host_de_mask, cpu_out2, width, height, MASK_SIZE);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of CPU opening: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_dil, cpu_out, width, height, "opening");
#endif

// Rotate

	float rotation_Rate = 0.70710678;//0.70710678

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif
	// Change width and height for new block and grid dim due to rotation
	int tmp = width;
	width = height;
	height = tmp;
	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/BLOCK_SIZE)+1;
	grid_dim.y = ((height-1)/BLOCK_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;

	rotate_gather_90<<<grid_dim, block_dim>>>(device_dil, device_rt, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of GPU rotate: %.6f seconds\n", milliseconds*1e-3);
#endif

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	err = cudaMemcpy(host_rt, device_rt, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	// Get intermediate data to test noise removal kernel
	if(err!=cudaSuccess){
		cout << "Failed to copy rs image to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	total_time += milliseconds;
	printf("Runtime of Device2Host memcpy: %.6f seconds\n", milliseconds*1e-3);
#endif

#ifdef RUN_TESTS
	cudaEventRecord(start);
	rotate_cpu(cpu_out, host_dil, width, height, -1, 0);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of CPU rotate: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_rt, cpu_out, width, height, "rotate");
#endif

// Rescaling

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	const unsigned int new_width  = roundf(width * rescale);
	const unsigned int new_height = roundf(height * rescale);
	grid_dim.x = ((new_width-1)/BLOCK_SIZE)+1;
	grid_dim.y = ((new_height-1)/BLOCK_SIZE)+1;
	rescaling_bilin<<<grid_dim, block_dim>>>(device_rt, device_rs, width, height, rescale, new_width, new_height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of GPU rescaling: %.6f seconds\n", milliseconds*1e-3);
#endif

	err = cudaMemcpy(host_rs, device_rs, new_width*new_height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	// Get intermediate data to test noise removal kernel
	if(err!=cudaSuccess){
		cout << "Failed to copy rs image to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaEventRecord(start);
	rescaling_bilin_cpu(cpu_out, host_rt, width, height, rescale);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of CPU rescale: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_rs, cpu_out, width, height, "rescaling");
#endif

	/* Add other cuda calls using images from previous kernel outputs */

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif
	/* Cuda mem copy for rgb input image */
	err = cudaMemcpy(device_rgb, rgb_image, width*height*rgb*sizeof(uint8_t), cudaMemcpyHostToDevice);
	if(err!=cudaSuccess){
		cout << "Failed to copy input image to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

	// Use same block and grid dim for noise removal kernel
	block_size = BLOCK_SIZE;
	grid_dim.x = ((width-1)/TILE_SIZE)+1;
	grid_dim.y = ((height-1)/TILE_SIZE)+1;
	grid_dim.z = 1;
	block_dim.x = block_size;
	block_dim.y = block_size;
	block_dim.z = 1;
	tiled_rgb2gray_inversion_noise_removal_binarization<<<grid_dim, block_dim>>>(device_rgb, device_gt, width, height);
	cudaDeviceSynchronize(); // Make sure eveything is completed

#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Runtime of all kernels on GPU: %.6f seconds\n", total_time*1e-3);
	printf("Runtime of GPU combined: %.6f seconds\n", milliseconds*1e-3);
#endif

	// Get intermediate data to test noise removal kernel
	err = cudaMemcpy(host_combined, device_gt, width*height*sizeof(uint8_t), cudaMemcpyDeviceToHost);
	if(err!=cudaSuccess){
		cout << "Failed to copy input image to device at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free nr output from device memory before failing
		cudaFree(device_nr_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_de_mask); // Free nr_mask output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		cudaFree(device_threshold); // Free threshold from device memory before failing
		return -1;
	}

#ifdef RUN_TESTS
	cudaEventRecord(start);
	binarization_cpu(cpu_out, host_nr, width, height);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("Runtime of CPU combined: %.6f seconds\n", milliseconds*1e-3);
	error_check(host_combined, cpu_out, width, height, "combined");
#endif


	/* Write image using stbi
	 * Image Name: Name of the output image
	 * Width: Width of the image
	 * Height: Height of the image
	 * Components: Components of the input image
	 *     N=#comp     components	
	 *		1           grey	
	 *		2           grey, alpha	
	 *		3           red, green, blue	
	 *		4           red, green, blue, alpha
	 * Output: Array for the output image
	 * Stride: Stride for the pixels in required in png write. Since we are using grayscale image we have a single channel with width stride
	 */
	// Swap back for easy image writes
	tmp = width;
	width = height;
	height = tmp;
	stbi_write_png("output_image_gsi.png", width, height, 1, host_inversion, width*1);
	stbi_write_png("output_image_nr.png", width, height, 1, host_nr, width*1);
	stbi_write_png("output_image_bin.png", width, height, 1, host_bin, width*1);
	stbi_write_png("output_image_combined.png", width, height, 1, host_combined, width*1);
	stbi_write_png("output_image_de.png", width, height, 1, host_dil, width*1);
	// return to rotated image dimensions
	tmp = width;
	width = height;
	height = tmp;
	stbi_write_png("output_image_rt.png", width, height, 1, host_rt, width*1);
	stbi_write_png("output_image_rs.png", new_width, new_height, 1, host_rs, new_width*1);

#ifdef RUN_TESSERACT
	api->SetImage(rgb_image, width, height, 3, width*3);
	//api->SetImage(host_inversion, width, height, 1, width*1);
    outText = api->GetUTF8Text();
    printf("OCR output original img:\n%s", outText);
	api->SetImage(host_rs, new_width, new_height, 1, new_width*1);
    outText = api->GetUTF8Text();
    printf("OCR output:\n%s", outText);
    api->End();
    delete api;
    delete [] outText;
#endif

	// Free everything
	stbi_image_free(rgb_image);
	free(host_inversion);
	free(host_nr);
	free(host_bin);
	free(host_combined);
	free(host_dil);
	free(host_ero);
	cudaFree(device_rgb); // Free rgb from device memory
	cudaFree(device_gs); // Free gs output from device memory
	cudaFree(device_nr); // Free nr output from device memory
	cudaFree(device_bin); // Free nr output from device memory
	cudaFree(device_nr_mask); // Free nr_mask output from device memory
	cudaFree(device_gt); // Free gt output from device memory
	cudaFree(device_de_mask); // Free nr_mask output from device memory
	cudaFree(device_dil); // Free dil output from device memory
	cudaFree(device_ero); // Free ero output from device memory
	cudaFree(device_rs); // Free rs output from device memory
	cudaFree(device_rt); // Free rt output from device memory
	cudaFree(device_hist); // Free hist output from device memory
	cudaFree(device_threshold); // Free threshold from device memory

	return 0;
}
