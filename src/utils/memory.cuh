#pragma once
#include "cuda.cuh"
#include <memory>
#include <string>
#include <stdexcept>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

//*****************************************************************************
//Definitions
//*****************************************************************************
namespace utils {
//Define some unique pointers, maybe create a structure to encapsulate size later
template <typename T>
using ArrayDeleter = std::default_delete<T[]>;

template <typename T, template<typename> typename Deleter>
using UniqueArrayPtr = std::unique_ptr<T[], Deleter<T>>;

template <typename T>
using Ptr = T *;

template <typename T>
using ConstPtr = const T *;

template <typename T>
Ptr<T> cudaAllocator(size_t length);

template <typename T>
struct CudaDeleter
{  
    //Forward declaration is impossible
    void operator()(Ptr<T> ptr) const
    {
        HANDLE_ERROR(cudaFree(ptr));
    }
};

template <typename T>
void cudaHostToDevice(ConstPtr<T> host, Ptr<T> device, uint32 nbElem);

template <typename T>
void cudaDeviceToHost(ConstPtr<T> device, Ptr<T> host, uint32 nbElem);

template<typename T>
void cudaFill(Ptr<T> device, int32 byteValue, uint32 nbElem);

template<typename T>
using DUniqueArrayPtr = UniqueArrayPtr<T, CudaDeleter>;

template<typename T>
using HUniqueArrayPtr = UniqueArrayPtr<T, ArrayDeleter>;

template<typename T>
using DUniquePtr = std::unique_ptr<T, CudaDeleter<T>>;

template<typename T>
using HUniquePtr = std::unique_ptr<T>;

template<typename T>
using HVector = thrust::host_vector<T>;

template<typename T>
using DVector = thrust::device_vector<T>;

template<typename T>
using DPtr = thrust::device_ptr<T>;
}
//*****************************************************************************
//Declarations
//*****************************************************************************
namespace utils {
//*****************************************************************************
template <typename T>
Ptr<T> cudaAllocator(size_t length)
{
    Ptr<T> ptr;
    HANDLE_ERROR(cudaMalloc((void**)&ptr, length*sizeof(T)));
    return ptr;
}
//*****************************************************************************
template <typename T>
void cudaHostToDevice(ConstPtr<T> host, Ptr<T> device, uint32 nbElem)
{
    HANDLE_ERROR(cudaMemcpy(device, host, nbElem*sizeof(T), cudaMemcpyHostToDevice));
}
//*****************************************************************************
template <typename T>
void cudaDeviceToHost(ConstPtr<T> device, Ptr<T> host, uint32 nbElem)
{
    HANDLE_ERROR(cudaMemcpy(host, device, nbElem*sizeof(T), cudaMemcpyDeviceToHost));
}
//*****************************************************************************
template<typename T>
void cudaFill(Ptr<T> device, int32 byteValue, uint32 nbElem)
{
    HANDLE_ERROR(cudaMemset(device, byteValue, nbElem*sizeof(T))); 
}
}
