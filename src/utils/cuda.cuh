#pragma once
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>
#include "define.h"

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define GLOBAL __global__
#else
#define HOST
#define DEVICE
#define GLOBAL
#endif
#define HANDLE_ERROR(_exp) do {											\
    const cudaError_t err = (_exp);										\
    if ( err != cudaSuccess ) {											\
        std::cerr	<< cudaGetErrorString( err ) << " in " << __FILE__	\
					<< " at line " << __LINE__ << std::endl;			\
        exit( EXIT_FAILURE );											\
    }																	\
} while (0)

namespace utils {
class KernelLauncher {
public:
    KernelLauncher();
    virtual void operator()() const = 0;
    static const cudaDeviceProp& deviceProp();
    static void initialize();
private:
    static bool mInitialized;
    static int32 mDevice;
    static cudaDeviceProp mProp;
};
int32 chooseBestDevice();
void cudaSafe(const char* errorMessage, const char*file=__FILE__, const int32 line=__LINE__);
}
