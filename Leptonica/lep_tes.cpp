#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>
#include <sys/time.h>
#include <stdio.h>

#define RUN_TESSERACT 1
#define WRITE_IMAGE 1

int main(int argc, char **argv){

#if RUN_TESSERACT
	char *outText;
	tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
	if (api->Init(NULL, "eng")) {
		 exit(1);
	}
#endif
	struct timeval start, end;
	char *fileName;
	PIX *pixC, *pixG, *pixINV;
	PIX *pixNR, *pixTH;
	PIX *pixO;
	PIX *pixRT,*pixRS;

	SEL *sel;
	sel = selCreate(3,3,"ALL_HITS");
	for(int i = 0; i<3; i++){
		for(int j = 0; j<3; j++){
			selSetElement(sel,i,j,SEL_HIT);
		}
	}

	fileName = argv[1];

	// Read input RGB image
	pixC = pixRead(fileName);
#if WRITE_IMAGE
	pixWrite("0_leptonica_original.png", pixC, IFF_PNG);
#endif

	// Start timer
	gettimeofday(&start, NULL);

	// Convert RGB to Gray
	pixG = pixConvertRGBToGray(pixC, 0.21, 0.71, 0.08);
	pixINV = pixInvert(pixG, pixG);
#if WRITE_IMAGE
	pixWrite("1_leptonica_grayscale_inverted.png", pixINV, IFF_PNG);
#endif

	// Apply noise removal with a 3x3 mask
	pixNR = pixBlockconvGrayTile(pixINV,NULL,1,1);
#if WRITE_IMAGE
	pixWrite("2_leptonica_noiseremoval.png", pixNR, IFF_PNG);
#endif

	// Apply adaptive otsu thresholding
	pixOtsuAdaptiveThreshold(pixNR, 5000, 5000, 0, 0, 0.0, NULL, &pixTH);
#if WRITE_IMAGE
	pixWrite("3_leptonica_threshold.png", pixTH, IFF_PNG);
#endif

	// Apply opening
	pixO = pixOpen(pixTH,pixTH,sel);
#if WRITE_IMAGE
	pixWrite("4_leptonica_opening.png", pixO, IFF_PNG);
#endif

	// Rotate 90 degrees
	pixRT = pixRotate90(pixO, -1);
#if WRITE_IMAGE
	pixWrite("5_leptonica_rotate.png", pixRT, IFF_PNG);
#endif

	// Scale by 2.5
	pixRS = pixScaleBinary(pixRT, 2.5, 2.5);
#if WRITE_IMAGE
	pixWrite("6_leptonica_rescale.png", pixRS, IFF_PNG);
#endif

	// End timer
	gettimeofday(&end, NULL);
	long seconds = (end.tv_sec - start.tv_sec);
	long micro = ((seconds * 1000000) + end.tv_usec) - start.tv_usec;
	printf("Total Runtime:%.3f ms\n", micro*0.001);

#if RUN_TESSERACT
	/*
	 * Start Tesseract Part!
	 */
	printf("==================================================================================\n");
	api->SetImage(pixC);
	outText = api->GetUTF8Text();
	printf("OCR output original img:\n%s", outText);
	printf("==================================================================================\n");
	api->SetImage(pixRS);
	outText = api->GetUTF8Text();
	printf("OCR output after pre-processing:\n%s", outText);
	printf("==================================================================================\n");
	api->End();
	delete api;
	delete [] outText;
#endif

	pixDestroy(&pixC);
	pixDestroy(&pixG);
//	pixDestroy(&pixINV);
	pixDestroy(&pixNR);
	pixDestroy(&pixTH);
//	pixDestroy(&pixO);
	pixDestroy(&pixRT);
	pixDestroy(&pixRS);
	selDestroy(&sel);

	return 0;
}
