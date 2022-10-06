CXX = nvcc
CXXFLAGS = -w -lm
TFLAGS = -std=c++17 -ltesseract -DRUN_TESSERACT

INCS=$(HOME)/local/include/
LIBS=$(HOME)/local/lib/

default: main.cu kernels.cu
	$(CXX) -I$(INCS) -L$(LIBS) $(CXXFLAGS) $(TFLAGS) main.cu -o image_processing

stream: streaming_main.cu
	 $(CXX) $(CXXFLAGS) -DTIME_CUDA --default-stream per-thread -DTIME_CUDA streaming_main.cu -o image_processing

stream_tesseract: streaming_main.cu
	 $(CXX) -I$(INCS) -L$(LIBS) $(CXXFLAGS) $(TFLAGS) streaming_main.cu -o image_processing

test_cpu: main.cu kernels.cu cpu_functions.cpp
	$(CXX) $(CXXFLAGS) -DRUN_TESTS main.cu cpu_functions.cpp -o image_processing

time_gpu: main.cu kernels.cu
	$(CXX) $(CXXFLAGS) -DTIME_CUDA main.cu -o image_processing

test: main.cu kernels.cu cpu_functions.cpp
	$(CXX) $(CXXFLAGS) -DRUN_TESTS -DTIME_CUDA main.cu cpu_functions.cpp -o image_processing

test_all: main.cu kernels.cu cpu_functions.cpp
	$(CXX) $(CXXFLAGS) $(TFLAGS) -DRUN_TESTS -DTIME_CUDA main.cu cpu_functions.cpp -o image_processing

memory_info: main.cu kernels.cu cpu_functions.cpp
	$(CXX) $(CXXFLAGS) -DRUN_TESTS -DTIME_CUDA main.cu -Xptxas="-v" cpu_functions.cpp -o image_processing
	
clean:
	rm -f image_processing
