#pragma once
#include <thrust/functional.h>
#include "cuda.cuh"
#include "memory.cuh"
#include "vec.cuh"

namespace utils {
//*****************************************************************************
class OffsetFunctor : public thrust::binary_function<uint32,uint32,uint32>
{
public:
    OffsetFunctor(uint32 const * const cellsStop);
    HOST DEVICE uint32 operator()(uint32 index, uint32 offset);
private:
    uint32 const * const cellsStop;
};
//*****************************************************************************
class IntegrateFunctor : public thrust::binary_function<Vec2f, Vec2f, Vec2f>
{
public:
    IntegrateFunctor(float dt);
    HOST DEVICE Vec2f operator()(const Vec2f& left, const Vec2f& right);
private:
    float dt;
};
//*****************************************************************************
class NbPartFunctor : public thrust::unary_function<float, uint32>
{
public:
    NbPartFunctor(uint32 nbIte, float intensityCoeff);
    HOST DEVICE uint32 operator()(float val);
private:
    uint32 nbIte;
    float intensityCoeff;
};
//*****************************************************************************
struct MinFunctor : public thrust::binary_function<Vec2d, Vec2d, Vec2d>
{
    HOST DEVICE Vec2d operator()(const Vec2d& left, const Vec2d& right);
};
//*****************************************************************************
struct MaxFunctor : public thrust::binary_function<Vec2d, Vec2d, Vec2d>
{
    HOST DEVICE Vec2d operator()(const Vec2d& left, const Vec2d& right);
};
//*****************************************************************************
class DivideAvgFunctor : public thrust::binary_function<Vec2d, uint32, Vec2d>
{
public:
    DivideAvgFunctor(ConstPtr<uint32> nbElems);
    HOST DEVICE Vec2d operator()(const Vec2d& left, uint32 right);
private:
    ConstPtr<uint32> nbElems;
};
}

