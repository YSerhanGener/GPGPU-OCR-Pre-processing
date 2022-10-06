#ifdef RUN_TESSERACT
#include <tesseract/baseapi.h>
#endif

#include <string>
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

#if defined(TIME_CUDA)
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	float total_time = 0;
#endif

	int tmp;
	int rgb = 3;
	char imagepath[1024];
	cudaStream_t streams[30];
	if (argc < 2 || argc > 31) {
		cout << "Usage: " << argv[0] << " [image file]..." << endl;
		return -1;
	}

	unsigned int num_images = argc - 1;

	for (unsigned int i = 0; i < num_images; i++) {
		cudaStreamCreate(&(streams[i]));
	}
	/* Read image from using stbi
	 * Image Name: Path to image file
	 * Width: Width of the image
	 * Height: Height of the image
	 * Components: Components of the input image
	 *		 N=#comp		 components	
	 *		1					 grey	
	 *		2					 grey, alpha	
	 *		3					 red, green, blue	
	 *		4					 red, green, blue, alpha
	 * RGB: Requested components
	 */
	uint8_t **rgb_image = (uint8_t **) malloc(num_images * sizeof(uint8_t*));
	uint8_t *input_image;
	int *width = (int *) malloc(num_images * sizeof(int));
	int *height = (int *) malloc(num_images * sizeof(int));
	int *components = (int *) malloc(num_images * sizeof(int));
	
	const float rescale = 2;
	size_t *offsets = (size_t *) malloc(num_images * sizeof(size_t));
	size_t *rescaleOffsets = (size_t *) malloc(num_images * sizeof(size_t));
	int *numPixels = (int *) malloc(num_images * sizeof(int));
	int *rescaleNumPixels = (int *) malloc(num_images * sizeof(int));
	size_t totalPixels = 0;
	size_t rescaleTotalPixels = 0;

	for (unsigned int i = 0; i < num_images; i++) {
		strcpy(imagepath, argv[i + 1]);
		cout << "Reading image located at " << imagepath << endl;
		rgb_image[i] = stbi_load(imagepath, &width[i], &height[i], &components[i], rgb);
		if (rgb_image[i] == NULL) {
			cout << "Failed to open image " << imagepath << endl;
			return -1;
		}
		cout << "Image Width:" << width[i] << "\tHeight:" << height[i] << "\t#ofChannels:" << components[i] << std::endl;
		if (i) {
			offsets[i] = offsets[i - 1] + numPixels[i - 1];
			rescaleOffsets[i] = rescaleOffsets[i - 1] + rescaleNumPixels[i - 1];
		} else {
			offsets[i] = 0;
			rescaleOffsets[i] = 0;
		}
		numPixels[i] = width[i] * height[i];
		rescaleNumPixels[i] = roundf(width[i] * rescale) * roundf(height[i] * rescale);
	}
	totalPixels += offsets[num_images - 1] + numPixels[num_images - 1];
	rescaleTotalPixels += rescaleOffsets[num_images - 1] + rescaleNumPixels[num_images - 1];
	
	cudaHostAlloc((void**)&input_image, totalPixels * rgb * sizeof(uint8_t), cudaHostAllocDefault);

	for (unsigned int i = 0; i < num_images; i++) {
		for(unsigned int j = 0; j < numPixels[i]*rgb; j++){
			input_image[(offsets[i]*rgb)+j] = rgb_image[i][j];
		}
	}
 
 	/* decleration of variables */
	uint8_t *device_rgb, *device_gs, *device_nr, *device_bin, *device_gt, *device_dil, *device_ero, *device_rs, *device_rt;
	uint8_t *host_rs;
	unsigned int *device_hist;
	uint8_t *device_threshold, *host_threshold;

	const float host_nr_mask[MASK_DIM] = {1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM, 1.0/MASK_DIM,1.0/MASK_DIM,1.0/MASK_DIM};
	const uint8_t host_de_mask[MASK_DIM] = {1,1,1,1,1,1,1,1,1};

	cudaHostAlloc((void**) &host_rs, rescaleTotalPixels * sizeof(uint8_t), cudaHostAllocDefault);
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
	err = cudaMalloc((void **)&device_rgb, totalPixels * rgb * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for input image at line " << __LINE__ << endl;
		return -1;
	}
	err = cudaMalloc((void **)&device_gs, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for gs image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		return -1;
	}
	/* Cuda malloc for noise removal output and mask*/
	err = cudaMalloc((void **)&device_nr, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for nr image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_bin, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for binarization image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		return -1;
	}
	/* Cuda malloc for global thresholding*/
	err = cudaMalloc((void **)&device_gt, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for gt image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		return -1;
	}
	/* Cuda malloc for dilation*/
	err = cudaMalloc((void **)&device_dil, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for dil image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		return -1;
	}
	/* Cuda malloc for erosion*/
	err = cudaMalloc((void **)&device_ero, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for ero image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		return -1;
	}
	/* Cuda malloc for rescale*/
	err = cudaMalloc((void **)&device_rs, rescaleTotalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for rs image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		return -1;
	}
	/* Cuda malloc for rotate*/
	err = cudaMalloc((void **)&device_rt, totalPixels * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for rt image at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		return -1;
	}
	/* Cuda malloc for adaptive thresholding*/
	err = cudaMalloc((void **)&device_hist,  (NUM_BINS*num_images) * sizeof(uint8_t));
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for histogram at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		return -1;
	}
	err = cudaMalloc((void **)&device_threshold, sizeof(uint8_t) * num_images); // unsigned int
	if(err!=cudaSuccess){
		cout << "Failed to allocate device memory for threshold at line " << __LINE__ << endl;
		cudaFree(device_rgb); // Free rgb from device memory before failing
		cudaFree(device_gs); // Free gs output from device memory before failing
		cudaFree(device_nr); // Free nr output from device memory before failing
		cudaFree(device_bin); // Free bin output from device memory before failing
		cudaFree(device_gt); // Free gt output from device memory before failing
		cudaFree(device_dil); // Free dil output from device memory before failing
		cudaFree(device_ero); // Free ero output from device memory before failing
		cudaFree(device_rs); // Free rs output from device memory before failing
		cudaFree(device_rt); // Free rt output from device memory before failing
		cudaFree(device_hist); // Free hist output from device memory before failing
		return -1;
	}

	// Initilize block and grid configuration

#ifdef TIME_CUDA
	cudaEventRecord(start);
#endif

	int block_size = BLOCK_SIZE;
	dim3 grid_dim(((width[0]-1)/block_size)+1,((height[0]-1)/block_size)+1,1);
	dim3 block_dim(block_size,block_size,1);

	for (unsigned int i = 0; i < num_images; i++) {
		/* Cuda mem copy for rgb input image */
		err = cudaMemcpyAsync(device_rgb + (offsets[i]*rgb), input_image + (offsets[i]*rgb), numPixels[i] * rgb * sizeof(uint8_t), cudaMemcpyHostToDevice, streams[i]);
		if(err!=cudaSuccess){
			cout << "Failed to copy input image to device at line " << __LINE__ << endl;
			goto fail_exit;
		}
//	}

//	for (unsigned int i = 0; i < num_images; i++) {

		// Set block and grid dim for rgb to gray+inversion kernel
		block_size = BLOCK_SIZE;
		grid_dim.x = ((width[i]-1)/block_size)+1;
		grid_dim.y = ((height[i]-1)/block_size)+1;
		grid_dim.z = 1;
		block_dim.x = block_size;
		block_dim.y = block_size;
		block_dim.z = 1;

		// Call rgb to gray+inversion kernel
		rgb2gray_invesion<<<grid_dim, block_dim, 0, streams[i]>>>(device_rgb + (offsets[i]*rgb), device_gs + offsets[i], width[i], height[i]);

		// Tiled Noise Removal
		
		//  Set block and grid dim for noise removal kernel
		block_size = BLOCK_SIZE;
		grid_dim.x = ((width[i]-1)/TILE_SIZE)+1;
		grid_dim.y = ((height[i]-1)/TILE_SIZE)+1;
		grid_dim.z = 1;
		block_dim.x = block_size;
		block_dim.y = block_size;
		block_dim.z = 1;
		tiled_noise_removal<<<grid_dim, block_dim, 0, streams[i]>>>(device_gs + offsets[i], device_nr + offsets[i], width[i], height[i]);
		//tiled_rgb2gray_inversion_noise_removal<<<grid_dim, block_dim, 0, streams[i]>>>(device_rgb + (offsets[i]*rgb), device_nr + offsets[i], width[i], height[i]);
		//tiled_rgb2gray_inversion_noise_removal_histogram<<<grid_dim, block_dim, 0, streams[i]>>>(device_rgb + (offsets[i]*rgb), device_nr + offsets[i], width[i], height[i], device_hist + (NUM_BINS*i), numPixels[i]);
		// Binarization

		// Set block and grid dim for adaptive binarization kernel
		block_size = 1024;
		grid_dim.x = (numPixels[i]-1)/block_size +1;
		grid_dim.y = 1;
		grid_dim.z = 1;
		block_dim.x = block_size;
		block_dim.y = 1;
		block_dim.z = 1;


		adaptive_threshold_histogram<<<grid_dim,block_dim, 0, streams[i]>>>(device_hist + (NUM_BINS*i), device_nr + offsets[i], numPixels[i]);
		adaptive_threshold_calculation<<<1,NUM_BINS, 0, streams[i]>>>(device_hist + (NUM_BINS*i), numPixels[i], device_threshold + i);
		adaptive_threshold_apply<<<grid_dim,block_dim, 0, streams[i]>>>(device_nr + offsets[i], device_bin + offsets[i], numPixels[i], device_threshold + i);

		//	 Opening - Dilation and Erosion

		block_size = BLOCK_SIZE;
		grid_dim.x = ((width[i]-1)/TILE_SIZE)+1;
		grid_dim.y = ((height[i]-1)/TILE_SIZE)+1;
		//grid_dim.x = ((width[i]-1)/LOW_TILE_SIZE)+1;
		//grid_dim.y = ((height[i]-1)/LOW_TILE_SIZE)+1;
		grid_dim.z = 1;
		block_dim.x = block_size;
		block_dim.y = block_size;
		block_dim.z = 1;
		shared_erosion<<<grid_dim, block_dim, 0, streams[i]>>>(device_bin + offsets[i], device_ero + offsets[i], width[i], height[i]);
		shared_dilation<<<grid_dim, block_dim, 0, streams[i]>>>(device_ero + offsets[i], device_dil + offsets[i], width[i], height[i]);
		//shared_opening<<<grid_dim, block_dim, 0, streams[i]>>>(device_bin + offsets[i], device_dil + offsets[i], width[i], height[i]);

		// Change width and height for new block and grid dim due to rotation
		tmp = width[i];
		width[i] = height[i];
		height[i] = tmp;
		block_size = BLOCK_SIZE;
		grid_dim.x = ((width[i]-1)/BLOCK_SIZE)+1;
		grid_dim.y = ((height[i]-1)/BLOCK_SIZE)+1;
		grid_dim.z = 1;
		block_dim.x = block_size;
		block_dim.y = block_size;

		// Rotate

		//rotate_gather<<<grid_dim, block_dim, 0, streams[i]>>>(device_rs + offsets[i], device_rt + offsets[i], width[i], height[i], -1, 0);
		rotate_gather_90<<<grid_dim, block_dim, 0, streams[i]>>>(device_dil + offsets[i], device_rt + offsets[i], width[i], height[i]);

		// Rescaling

		const unsigned int new_width  = roundf(width[i] * rescale);
		const unsigned int new_height = roundf(height[i]  * rescale);
		grid_dim.x = ((new_width-1)/BLOCK_SIZE)+1;
		grid_dim.y = ((new_height-1)/BLOCK_SIZE)+1;
		rescaling_bilin<<<grid_dim, block_dim, 0, streams[i]>>>(device_rt + offsets[i], device_rs + rescaleOffsets[i], width[i], height[i], rescale, new_width, new_height);

//}

//		for (unsigned int i = 0; i < num_images; i++) {
//			cout << i << endl;
//			cudaStreamSynchronize(streams[i]);
//			cout << "Stream Sync" << cudaGetErrorName(cudaGetLastError()) << endl;
//		}

//	for (unsigned int i = 0; i < num_images; i++) {

		err = cudaMemcpyAsync(host_rs + rescaleOffsets[i], device_rs + rescaleOffsets[i], rescaleNumPixels[i] * sizeof(uint8_t), cudaMemcpyDeviceToHost, streams[i]);
		//err = cudaMemcpy(host_rt, device_rt, numPixels[0] * sizeof(uint8_t), cudaMemcpyDeviceToHost);
//			cout << "memcpy2" << cudaGetErrorName(cudaGetLastError()) << endl;
		// Get intermediate data to test noise removal kernel
		if(err!=cudaSuccess){
			cout << "Failed to copy rs image to device at line " << __LINE__ << endl;
			goto fail_exit;
		}

	}

	for (unsigned int i = 0; i < num_images; i++) {
		cudaStreamSynchronize(streams[i]);
	}
#ifdef TIME_CUDA
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Number of images:%d\n",num_images);
	printf("%.3f\n", milliseconds/num_images);
#endif

	/* Write image using stbi
	 * Image Name: Name of the output image
	 * Width: Width of the image
	 * Height: Height of the image
	 * Components: Components of the input image
	 *		 N=#comp		 components	
	 *		1					 grey	
	 *		2					 grey, alpha	
	 *		3					 red, green, blue	
	 *		4					 red, green, blue, alpha
	 * Output: Array for the output image
	 * Stride: Stride for the pixels in required in png write. Since we are using grayscale image we have a single channel with width stride
	 */
	char str[1024];
	for (unsigned int i = 0; i < num_images; i++) {
		const unsigned int new_width  = roundf(width[i]  * rescale);
		const unsigned int new_height = roundf(height[i] * rescale);

		cudaStreamSynchronize(streams[i]);
		sprintf(str, "output_image_%d.png", i);
		stbi_write_png(str, new_width, new_height, 1, host_rs + rescaleOffsets[i], new_width);
		//stbi_write_png(str, height[i], width[i], 1, host_rs + offsets[i], height[i]);
		//stbi_write_png("input_rgb.png", height[i], width[i], 3, input_image + offsets[i], height[i]*3);
	}


#ifdef RUN_TESSERACT
	for (unsigned int i = 0; i < num_images; i++) {
	    	const unsigned int new_width  = roundf(width[i]  * rescale);
    		const unsigned int new_height = roundf(height[i] * rescale);

		sprintf(str, "output_image_%d.png", i);
		// Use swaped width height due to rotatation
		api->SetImage(input_image + offsets[i], height[i], width[i], 3, height[i]*3);
		outText = api->GetUTF8Text();
		printf("OCR output original img:\n%s", outText);
		api->SetImage(host_rs + rescaleOffsets[i], new_width, new_height, 1, new_width);
		outText = api->GetUTF8Text();
		printf("OCR output for %s:\n%s", str, outText);
		printf("==================================================================================\n");
		printf("Line and word segments:\n");
		api->SetImage(host_rs + rescaleOffsets[i], new_width, new_height, 1, new_width);
		api->Recognize(0);
		tesseract::ResultIterator* ri = api->GetIterator();
		tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
		if(ri != 0){
			do{
				const char* word = ri->GetUTF8Text(level);
				//float conf = ri->Confidence(level);
				int x1, x2, y1, y2;
				ri->BoundingBox(level, &x1, &x2, &y1, &y2);
				//printf("word: '%s'; \tconf: %.2f; BoundingBox: %d,%d,%d,%d;\n", word, conf, x1, x2, y1, y2);
				printf("word: '%s';  BoundingBox: %d,%d,%d,%d;\n", word, x1, x2, y1, y2);
				delete[] word;
			} while(ri->Next(level));
		}
		printf("==================================================================================\n");
	}
	api->End();
	delete api;
	delete [] outText;
#endif

	// Free everything
	stbi_image_free(rgb_image);
	cudaFreeHost(host_rs);
	cudaFreeHost(input_image);
	cudaFree(device_rgb); // Free rgb from device memory
	cudaFree(device_gs); // Free gs output from device memory
	cudaFree(device_nr); // Free nr output from device memory
	cudaFree(device_bin); // Free nr output from device memory
	cudaFree(device_gt); // Free gt output from device memory
	cudaFree(device_dil); // Free dil output from device memory
	cudaFree(device_ero); // Free ero output from device memory
	cudaFree(device_rs); // Free rs output from device memory
	cudaFree(device_rt); // Free rt output from device memory
	cudaFree(device_hist); // Free hist output from device memory
	cudaFree(device_threshold); // Free threshold from device memory

	for (unsigned int i = 0; i < num_images; i++) {
		cudaStreamDestroy(streams[i]);
	}

	return 0;

fail_exit:
	// Free everything
	stbi_image_free(rgb_image);
	cudaFreeHost(host_rs);
	cudaFreeHost(input_image);
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

	for (unsigned int i = 0; i < num_images; i++) {
		cudaStreamDestroy(streams[i]);
	}

#ifdef RUN_TESSERACT
	api->End();
	delete api;
#endif

	return -1;
}
