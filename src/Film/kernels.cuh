#pragma once
#include "utils/cuda.cuh"
#include "utils/vec.cuh"
#include "utils/memory.cuh"
#include <curand_kernel.h>
#include <thrust/pair.h>

namespace film {
//*****************************************************************************
GLOBAL void initSPH(utils::Ptr<float> intensities, utils::Ptr<utils::Vec2f> centers,
    utils::Ptr<utils::Vec2f> lowerLimits, utils::Ptr<utils::Vec2f> upperLimits,
    utils::Ptr<utils::Vec2f> pos, utils::Ptr<float> mass, utils::Ptr<float> radius,
    utils::Ptr<uint32> worldKey, utils::ConstPtr<utils::Vec2f> possiblePos,
    utils::ConstPtr<uint32> partStop, utils::Ptr<curandState> states, utils::Ptr<bool> mark,
    float intensityCoeff, float avgMass, float stdevMass, float rho0, utils::Vec2<uint32> dim, uint32 maxPart);
//*****************************************************************************
GLOBAL void initRandomPos(utils::Ptr<float> intensities, utils::Ptr<utils::Vec2f> centers,
    utils::Ptr<utils::Vec2f> lowerLimits, utils::Ptr<utils::Vec2f> upperLimits,
    utils::Ptr<curandState> states, utils::Ptr<bool> mark, utils::Ptr<uint32> nbPart,
    float intensityCoeff, utils::Vec2<uint32> dim, uint32 maxPart);
//*****************************************************************************
GLOBAL void initRandomSPH(utils::Ptr<utils::Vec2f> pos, utils::Ptr<float> mass,
    utils::Ptr<float> radius, utils::Ptr<uint32> worldKey, utils::ConstPtr<utils::Vec2f> possiblePos,
    utils::ConstPtr<uint32> partStop, utils::Ptr<curandState> states, utils::ConstPtr<bool> mark,
    float avgMass, float stdevMass, float rho0, utils::Vec2<uint32> dim, uint32 maxPart);
//*****************************************************************************
GLOBAL void createGrid(utils::ConstPtr<utils::Vec2f> pos, utils::Ptr<uint32> indexes,
    utils::Ptr<uint32> offset, utils::Ptr<uint32> partByCell, uint32 partNumber,
    utils::Vec2<uint32> dim, utils::Vec2f pixelSize, thrust::pair<uint32,uint32> firstPixel);
//*****************************************************************************
GLOBAL void computeDensity(utils::Ptr<float> densities, utils::ConstPtr<utils::Vec2f> pos,
    utils::ConstPtr<float> grainsRadius, utils::ConstPtr<uint32> cellsStop,
    thrust::pair<uint32, uint32> nbSamples, float maxGrainRadius, uint32 width,
    uint32 height, utils::Vec2f ratio, thrust::pair<uint32, uint32> firstPixel,
    bool debug);
//*****************************************************************************
GLOBAL void initRandomState(utils::Ptr<curandState> states, uint64 seed, uint32 nbPart);
}
