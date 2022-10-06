# ECE569_Project

## Makefile Usage
You can use `make` to build the gpu only version with tesseract and `make test_cpu` to build both gpu and cpu versions for correctness comparison. With `make time_gpu` enable timing calculations for gpu version. Running `make test` to build will make an executable that tests gpu-cpu versions and print timing results for all versions. `make test_all` will compile the source with all tests and tesseract enabled. Can also run `make memory_info` to display the kernel memory information for all the kernels. `make stream` will compile the source with streaming enabled. `make stream_tesseract` will compile the source with streaming enabled and run tesseract on all images at the end when running the executable.

`make stream` will compile a version that will be able to take several images and put them into streams.

## Executable Usage
If you call `./image_processing` it will take the image.png in the home directory as input. You can also call `./image_processing inputs/4_GS8homeScreen.jpg` to specify an input image.

#### Without streaming
```
./image_processing IMAGE
```
#### With streaming
```
./image_processing IMAGE1 IMAGE2...
```

