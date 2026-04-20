#include <string>
#include <iostream>

#include "functions.h"


int main() {

	int devices = deviceCount();
	if (!devices) {
		std::cout << "⚠️ No ROCM compatible devices found!" << std::endl;
		return 1;
	}
	std::cout << "✅ Found " << devices  <<" ROCM devices" << std::endl;

	int err = initGPU();
	if (err != 0) {
		std::cout << "⚠️ Issue setting ROCM device. Error: " << err << std::endl;
		return 1;
	}


	// pageableMemTest();
	// pinnedMemTest();

	// streamingTest();

	dotProductTest();



    return 0; 
}
