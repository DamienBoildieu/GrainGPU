#pragma once
#include "utils/cuda.cuh"
#include "utils/memory.cuh"
#include "utils/vec.cuh"
#include <thrust/pair.h>
#include <curand_kernel.h>

namespace film {
//*****************************************************************************
class InitKernel : public utils::KernelLauncher
{
public:
    InitKernel(utils::Ptr<float> intensities, utils::Ptr<utils::Vec2f> centers,
        utils::Ptr<utils::Vec2f> lowerLimits, utils::Ptr<utils::Vec2f> upperLimits,
        utils::Ptr<utils::Vec2f> pos, utils::Ptr<float> mass, utils::Ptr<float> radius,
        utils::Ptr<uint32> worldKey, utils::ConstPtr<utils::Vec2f> possiblePos,
        utils::ConstPtr<uint32> partStop, utils::Ptr<curandState> states,
        utils::Ptr<bool> mark, float intensityCoeff, float avgMass, float stdevMass,
        float rho0, const utils::Vec2<uint32>& dim, uint32 partMax);
    void operator()() const override;
private:
    utils::Ptr<float> intensities;
    utils::Ptr<utils::Vec2f> centers;
    utils::Ptr<utils::Vec2f> lowerLimits;
    utils::Ptr<utils::Vec2f> upperLimits;
    utils::Ptr<utils::Vec2f> pos;
    utils::Ptr<float> mass;
    utils::Ptr<float> radius;
    utils::Ptr<uint32> worldKey;
    utils::ConstPtr<utils::Vec2f> possiblePos;
    utils::ConstPtr<uint32> partStop;

    utils::Ptr<curandState> states;
    utils::Ptr<bool> mark;
    float intensityCoeff;
    float avgMass;
    float stdevMass;
    float rho0;
    utils::Vec2<uint32> dim;
    uint32 partMax;

};
//*****************************************************************************
class RandomPosKernel : public utils::KernelLauncher
{
public:
    RandomPosKernel(utils::Ptr<float> intensities, utils::Ptr<utils::Vec2f> centers,
        utils::Ptr<utils::Vec2f> lowerLimits, utils::Ptr<utils::Vec2f> upperLimits,
        utils::Ptr<curandState> states, utils::Ptr<bool> mark, utils::Ptr<uint32> nbPart,
        float intensityCoeff, const utils::Vec2<uint32>& dim, uint32 maxPart);
    void operator()() const override;
private:
    utils::Ptr<float> intensities;
    utils::Ptr<utils::Vec2f> centers;
    utils::Ptr<utils::Vec2f> lowerLimits;
    utils::Ptr<utils::Vec2f> upperLimits;
    utils::Ptr<curandState> states;
    utils::Ptr<bool> mark;
    utils::Ptr<uint32> nbPart;
    float intensityCoeff;
    utils::Vec2<uint32> dim;
    uint32 maxPart;
};
//*****************************************************************************
class InitRandomKernel : public utils::KernelLauncher
{
public:
    InitRandomKernel(utils::Ptr<utils::Vec2f> pos, utils::Ptr<float> mass,
        utils::Ptr<float> radius, utils::Ptr<uint32> worldKey,
        utils::ConstPtr<utils::Vec2f> possiblePos, utils::ConstPtr<uint32> partStop,
        utils::Ptr<curandState> states, utils::ConstPtr<bool> mark, float avgMass,
        float stdevMass, float rho0, const utils::Vec2<uint32>& dim, uint32 maxPart);
    void operator()() const override;
private:
    utils::Ptr<utils::Vec2f> pos;
    utils::Ptr<float> mass;
    utils::Ptr<float> radius;
    utils::Ptr<uint32> worldKey;
    utils::ConstPtr<utils::Vec2f> possiblePos;
    utils::ConstPtr<uint32> partStop;
    utils::Ptr<curandState> states;
    utils::ConstPtr<bool> mark;
    float avgMass;
    float stdevMass;
    float rho0;
    utils::Vec2<uint32> dim;
    uint32 maxPart;
};
//*****************************************************************************
class GridKernel : public utils::KernelLauncher
{
public:
    GridKernel(utils::ConstPtr<utils::Vec2f> pos, utils::Ptr<uint32> indexes,
        utils::Ptr<uint32> offset, utils::Ptr<uint32> partByCell, uint32 partNumber,
        const utils::Vec2<uint32>& gridDim, const utils::Vec2f& pixelSize,
        const thrust::pair<uint32,uint32>& firstPixel = {0u,0u});
    void operator()() const override;
private:
    utils::ConstPtr<utils::Vec2f> pos;
    utils::Ptr<uint32> indexes;
    utils::Ptr<uint32> offset;
    utils::Ptr<uint32> partByCell;
    uint32 partNumber;
    utils::Vec2<uint32> gridDim;
    utils::Vec2f pixelSize;
    thrust::pair<uint32,uint32> firstPixel;
};
//*****************************************************************************
class DensityKernel : public utils::KernelLauncher
{
public:
    DensityKernel(utils::Ptr<float> densities, utils::ConstPtr<utils::Vec2f> pos,
        utils::ConstPtr<float> grainsRadius, utils::ConstPtr<uint32> cellsStop,
        const thrust::pair<uint32, uint32>& nbSamples, float maxGrainRadius, uint32 width,
        uint32 height, const utils::Vec2f& ratio, const thrust::pair<uint32,uint32>& firstPixel = {0u,0u},
        bool debug = false);
    void operator()() const override;
private:
    utils::Ptr<float> densities;

    utils::ConstPtr<utils::Vec2f> pos;
    utils::ConstPtr<float> grainsRadius;

    utils::ConstPtr<uint32> cellsStop;
    thrust::pair<uint32, uint32> nbSamples;
    float maxGrainRadius;
    uint32 width;
    uint32 height;
    utils::Vec2f ratio;
    thrust::pair<uint32,uint32> firstPixel;
    
    bool debug;

};
//*****************************************************************************
class RandomKernel : public utils::KernelLauncher
{
public:
    RandomKernel(utils::Ptr<curandState> states, uint64 seed, uint32 nbStates);
    void operator()() const override;
private:
    utils::Ptr<curandState> states;
    uint64 seed;
    uint32 nbStates;
};
}
