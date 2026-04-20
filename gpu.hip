#include <hip/hip_runtime.h>
#include <iostream>
// #include <iomanip>
#include <cstring>
#include <unistd.h>
#include <chrono>
#include <algorithm>


#include "kernels.h"

// the hip function calls are working, but for some reason VSCode does not properly detect the header file
// Not sure how much time to invest sinking into this.

int deviceCount() {

	int count;
	hipError_t devices = hipGetDeviceCount(&count);
	if (devices == hipSuccess) {
		for (int i=0; i<count; i++) {
			hipDeviceProp_t prop;
			if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
				std::cout << "Device " << i << " : " << prop.name << std::endl;
			}
		}
	} else {
		std::cout << "Encountered Error: " << devices << std::endl;
	}

	return count;
}


/**
 * @brief Configures the run context to properly communicate with the (only one I have)
 * ROCM compatible GPU.
 */
int initGPU() {
	// Device 0 : Radeon RX 7900 XT
	hipError_t err;
	int setDeviceID;
	
	err = hipSetDevice(0);
	if (err) {
		std::cout << "⚠️ Error setting GPU. Error: " << err << std::endl;
		return err;
	}

	err = hipGetDevice(&setDeviceID);
	if (err){
		std::cout << "⚠️ Error fetching current GPU from context. Error: " << err << std::endl;
		return err;
	}

	if (setDeviceID != 0) {
		std::cout << "⚠️ GPU set to incorrect device: " << setDeviceID << std::endl;
		return 1;
	}

	std::cout << "✅ Initialized context to GPU 0" << std::endl;
	return 0;
}



// convenience macro to print error if we get one, wouldn't necessarily need this
#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess)                   \
    {                                          \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}


void dumpMem(void* pointer, size_t size) {

	char* ptr = (char*)pointer;

	std::cout << "===== printing memory at " << (void*) ptr << " =====" << std::endl;

	// std::cout << "byte: -----------------------------------" << std::endl;
	// for (size_t i = 0; i < size; i++) {
	// 	std::cout << std::hex << std::setw(2) << std::setfill('0') << (unsigned int)*(ptr + i) << " ";
	// }
	// std::cout << "\n\n";


	// std::cout << "char: -----------------------------------" << std::endl;
	// for (size_t i = 0; i < size; i+=sizeof(char)) {
	// 	std::cout << (char)*(ptr + i) << " ";
	// }
	// std::cout << "\n\n";


	std::cout << "int: ------------------------------------" << std::endl;
	for (size_t i = 0; i < size; i+=sizeof(int)) {
		std::cout << std::dec << (int)*(ptr + i) << " ";
	}

	std::cout << "\n=============================================\n\n";
}


void pageableMemTest() {

	std::cout << "▶️  Pageable Memory Test" << std::endl;
	const int element_number = 100;
	const int arraySize = element_number * sizeof(int);

    int *host_input, *host_output;

    // Host allocation
    host_input  = new int[element_number];
    host_output = new int[element_number];

    // Host data preparation
    for (int i = 0; i < element_number; i++) {
        host_input[i] = i;
    }
	std::memset(host_output, 0, arraySize);

 
    int *device_input, *device_output;
	

    // Device allocation
	std::cout << "Malloc-ing " << arraySize << " bytes for device_input" << std::endl;
    HIP_CHECK(hipMalloc((int **)&device_input,  arraySize));
	std::cout << "Malloc-ing " << arraySize << " bytes for device_output" << std::endl;
    HIP_CHECK(hipMalloc((int **)&device_output, arraySize));

    // Device data preparation`
    HIP_CHECK(hipMemcpy(device_input, host_input, arraySize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_output, 0, arraySize));
    // Run the kernel
    // ...

	dumpMem(device_input, arraySize);
	dumpMem(device_output, arraySize);

    // Free host memory
    delete[] host_input;
    delete[] host_output;

    // Free device memory
    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));

}


void pinnedMemTest() {
	std::cout << "▶️  Pinned Memory Test" << std::endl;

	const int element_number = 100;
	const int arraySize = element_number * sizeof(int);

    int *host_input, *host_output;
    // Host allocation
    HIP_CHECK(hipHostMalloc(&host_input, arraySize));
    HIP_CHECK(hipHostMalloc(&host_output, arraySize));

    // Host data preparation
    for (int i = 0; i < element_number; i++)
    {
        host_input[i] = i;
    }
    std::memset(host_output, 0, arraySize);

    int *device_input, *device_output;

    // Device allocation
    HIP_CHECK(hipMalloc(&device_input,  arraySize));
    HIP_CHECK(hipMalloc(&device_output, arraySize));

    // Device data preparation
    HIP_CHECK(hipMemcpy(device_input, host_input, arraySize, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(device_output, 0, arraySize));

    // Run the kernel
    // ...

	dumpMem(device_input, arraySize);
	dumpMem(device_output, arraySize);


    // Free host memory
    HIP_CHECK(hipFreeHost(host_input));
    HIP_CHECK(hipFreeHost(host_output));

    // Free device memory
    HIP_CHECK(hipFree(device_input));
    HIP_CHECK(hipFree(device_output));
}






void streamingTest() {
	std::cout << "▶️  Test Streams" << std::endl;

	// Data setup
	// 	Two arrays:
	// 		input array: [1, 2, 3, 4 ...]
	// 		output_aray: [0, 0, 0, 0 ...]

	const int element_number = 100;
	const int arraySize = element_number * sizeof(int);
    int *host_input, *host_output;
    HIP_CHECK(hipHostMalloc(&host_input, arraySize));
    HIP_CHECK(hipHostMalloc(&host_output, arraySize));

    for (int i = 0; i < element_number; i++) {
        host_input[i] = i;
    }
    std::memset(host_output, 0, arraySize);

	int *device_input, *device_output;
    HIP_CHECK(hipMalloc(&device_input,  arraySize));
    HIP_CHECK(hipMalloc(&device_output, arraySize));







	// Streams setup
	// 	Three streams:
	// 		htdStream: 		copies the data from the host (RAM) to the device (VRAM)
	// 		computeStream: 	performs Y = X * -1 
	//							to the int from device_input (X) and places the result (Y) into 
	//							the corresponding index in device_output
	// 		dthStream: 		copies the data from the device (VRAM) back to the host (RAM)

	int const batchSize = 20;
	hipStream_t htdStream, computeStream, dthStream;
	hipEvent_t htdDone, computeDone;

	HIP_CHECK(hipStreamCreate(&htdStream));
	HIP_CHECK(hipStreamCreate(&computeStream));
	HIP_CHECK(hipStreamCreate(&dthStream));

	HIP_CHECK(hipEventCreate(&htdDone));
	HIP_CHECK(hipEventCreate(&computeDone));



	for (int i = 0; i < element_number / batchSize; i++) {
		int offset = i * batchSize;

		// Enqueue transfer on transferStream
		HIP_CHECK(hipMemcpyAsync(device_input + offset, host_input + offset, batchSize * sizeof(int), hipMemcpyHostToDevice, htdStream));

		// Record an event on transferStream — fires when transfer completes
		HIP_CHECK(hipEventRecord(htdDone, htdStream));

		// Make computeStream wait for that event before launching kernel
		HIP_CHECK(hipStreamWaitEvent(computeStream, htdDone, 0));



		// Run the kernel
		negate_kernel<<<dim3(1), dim3(20), 0, computeStream>>>(i, &(device_input[offset]), batchSize, &(device_output[offset]));



		// Record an event on computeStream — fires when compute completes
		HIP_CHECK(hipEventRecord(computeDone, computeStream));


		// Make dthStream wait for compute to finish before transferring to the host
		HIP_CHECK(hipStreamWaitEvent(dthStream, computeDone, 0));
		HIP_CHECK(hipMemcpyAsync(host_output + offset, device_output + offset, batchSize * sizeof(int), hipMemcpyDeviceToHost, dthStream));


		std::cout << "🚀  Queued stream to process chunk: " << offset << " to " << offset + batchSize << std::endl;
	}


	// TODO: experiment with hipLaunchHostFunc maybe? should try to get this printing to be event-based rather than
	// having this sleep in here.
	sleep(1);
	dumpMem(host_output, arraySize);
}


long current_millis() {
	return (long)std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::system_clock::now().time_since_epoch()
		).count();
}

void dotProductTest() {

	hipDeviceProp_t prop;
	HIP_CHECK(hipGetDeviceProperties(&prop, 0));
	printf("-> Shared mem per block: %zu bytes\n", prop.sharedMemPerBlock);
	printf("-> Max threads per block:     %d\n", prop.maxThreadsPerBlock);


	//LLAMA 3 70B does operations on matricies of size ~ 8192 square. Lets start with that. 
	int A_ROWS = 8192;
	int A_COLS = 8192;
	int B_ROWS = A_COLS;
	int B_COLS = 8192;

	// initialize matricies with random floats between 0 and 1
	float *A, *B;
	HIP_CHECK(hipHostMalloc(&A, A_COLS * A_ROWS * sizeof(float)));
	HIP_CHECK(hipHostMalloc(&B, B_COLS * B_ROWS * sizeof(float)));

	for (long i=0; i < (A_COLS * A_ROWS); i++) {
        A[i] = (float)rand() / (float)RAND_MAX;
	}

	for (long i=0; i < (B_COLS * B_ROWS); i++) {
        B[i] = (float)rand() / (float)RAND_MAX;
	}


	// copy matricies to device memory
	float *A_DEVICE, *B_DEVICE, *RESULT;
	HIP_CHECK(hipMalloc(&A_DEVICE, A_ROWS * A_COLS));
    HIP_CHECK(hipMalloc(&B_DEVICE, A_COLS * B_COLS));
	HIP_CHECK(hipMalloc(&RESULT, A_ROWS * B_COLS));

	HIP_CHECK(hipMemcpy(A_DEVICE, A, A_COLS * A_ROWS * sizeof(float), hipMemcpyHostToDevice));
	HIP_CHECK(hipMemcpy(B_DEVICE, B, B_COLS * B_ROWS * sizeof(float), hipMemcpyHostToDevice));


	// ----------------- START -----------------
	long start_millis = current_millis();


	int threads = 256; // means 16 by 16 tile
	// std::min(prop.maxThreadsPerBlock, ??);
	int blocks = (A_ROWS * B_COLS) / threads;

	if (blocks - ((A_ROWS * B_COLS) / threads) > 0) {
		// if the tiling doesn't fully cover the matrix, add 1
		blocks += 1;
	}

	std::cout << "Running a kernel with Blocks: " << blocks << " and Threads: " << threads << std::endl;

	sleep(5);
	
	dot_product<<<dim3(blocks), dim3(threads), 0>>>
	(A_ROWS, A_COLS, B_ROWS, B_COLS, A, B, RESULT);



	HIP_CHECK(hipStreamSynchronize(0));

	long end_millis = current_millis();
	long duration = end_millis - start_millis;
	printf("Duration: %ld\n", duration);
	// ----------------- END -----------------





	// dot_product_tiled<<<dim3(blocks), dim3(threads), 0>>>
	// (A_ROWS, A_COLS, B_ROWS, B_COLS, 32, A, B, RESULT);

}
