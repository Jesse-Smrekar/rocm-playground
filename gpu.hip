#include <hip/hip_runtime.h>
#include <iostream>
// #include <iomanip>
#include <cstring>
#include <unistd.h>

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

