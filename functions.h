#include <string>

#define PRINT_H
void print(std::string str);


#define GPU_H
int deviceCount();
int initGPU();
void pageableMemTest();
void pinnedMemTest();
void streamingTest();
void dotProductTest();
