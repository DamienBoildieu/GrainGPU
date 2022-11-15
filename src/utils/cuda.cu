#include "cuda.cuh"

namespace utils {
//*****************************************************************************
bool KernelLauncher::mInitialized = false;
//*****************************************************************************
int32 KernelLauncher::mDevice = chooseBestDevice();
//*****************************************************************************
cudaDeviceProp KernelLauncher::mProp;
//*****************************************************************************
KernelLauncher::KernelLauncher()
{
    initialize();
}
//*****************************************************************************
const cudaDeviceProp& KernelLauncher::deviceProp()
{
    initialize();
    return mProp;
}
//*****************************************************************************
void KernelLauncher::initialize()
{
    if (!mInitialized) {
        HANDLE_ERROR(cudaSetDevice(mDevice));
        HANDLE_ERROR(cudaGetDeviceProperties(&mProp, mDevice));
        mInitialized = true;
    }
}
//*****************************************************************************
HOST DEVICE
float radius(float mass, float rho0)
{
    return 0.;//cbrt(60.f*mass/(4.f*pi*rho0));
}
//*****************************************************************************
int32 chooseBestDevice() {
	// Get number of CUDA capable devices
	int32 nbDev;
	HANDLE_ERROR( cudaGetDeviceCount( &nbDev ) );

	if ( nbDev == 0 ) {
		std::cerr << "Error: no CUDA capable device" << std::endl;
		exit( EXIT_FAILURE );
	}

	// Choose best device
	int32 currentDev	= 0;
	int32 bestDev		= -1;
	int32 bestMajor	= 0;
	cudaDeviceProp propDev;
	while ( currentDev < nbDev ) {
		HANDLE_ERROR( cudaGetDeviceProperties( &propDev, currentDev ) );
		if ( propDev.major > bestMajor ) {
			bestDev		= currentDev;
			bestMajor	= propDev.major;
		}
		++currentDev;
	}
	return bestDev;
}
//*****************************************************************************
void cudaSafe(const char* errorMessage, const char*file, const int32 line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
	  fprintf(stderr,
			  "%s(%i) : getLastCudaError() CUDA error :"
			  " %s : (%d) %s.\n",
			  file,line, errorMessage, static_cast<int32>(err),
			  cudaGetErrorName(err));
			  cudaDeviceReset();
			  exit(EXIT_FAILURE);  
	}
}
}
