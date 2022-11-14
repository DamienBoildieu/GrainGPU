#pragma once
#include "utils/cuda.cuh"
#include "utils/vec.cuh"
#include "utils/memory.cuh"
#include "particles.cuh"

namespace sph {
//*****************************************************************************
//Declarations
//*****************************************************************************
/**
 * @brief The World struct
 * Define the particles list, limits and external forces
 */
struct World {
    HostVectorParticles particles;
    utils::Vec2f lowerLimits;
    utils::Vec2f upperLimits;
    utils::Vec2d externalForce;
};

struct DWorlds;
/**
 * @brief The HWorlds struct
 * Elements allocated on host
 */
struct HWorlds {
    inline void reset();
    inline uint32 nbParts(uint32 worldBegin, uint32 worldEnd) const;
    inline uint32 partBegin(uint32 worldBegin) const;
    inline utils::HVector<uint32> partsBegin() const;
    inline void copyFromDevice(const DWorlds& device, uint32 hPartOffset, uint32 nbPart,
        uint32 dPartOffset=0u);
    inline void append(const HWorlds& other);
    inline void append(const World& world);

    HostVectorParticles particles;
    utils::HVector<uint32> particlesByWorld;
    utils::HVector<uint32> worldKey;
    utils::HVector<utils::Vec2f> lowerLimits;
    utils::HVector<utils::Vec2f> upperLimits;
    utils::HVector<utils::Vec2d> externalForces;
};
/**
 * @brief The DWorlds struct
 * Elements allocated on device
 */
struct DWorlds {
    inline void reset();
    inline void reserve(uint32 nbWorld, uint32 nbPart);
    inline void copyFromHost(const HWorlds& host, uint32 hWorldOffset, uint32 nbWorld,
    uint32 hPartOffset, uint32 nbPart, uint32 dWorldOffset=0u, uint32 dPartOffset=0u);
    inline uint32 particleDependSize() const;
    inline uint32 worldDependSize() const;

    DevicePtrParticles particles;
    utils::DUniqueArrayPtr<uint32> worldKey;
    utils::DUniqueArrayPtr<utils::Vec2f> lowerLimits;
    utils::DUniqueArrayPtr<utils::Vec2f> upperLimits;
    utils::DUniqueArrayPtr<utils::Vec2d> externalForces;
};
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace sph {
    //*************************************************************************
void HWorlds::reset()
{
    particles.reset();
    particlesByWorld.clear();
    particlesByWorld.shrink_to_fit();
    worldKey.clear();
    worldKey.shrink_to_fit();
    lowerLimits.clear();
    lowerLimits.shrink_to_fit();
    upperLimits.clear();
    upperLimits.shrink_to_fit();
    externalForces.clear();
    externalForces.shrink_to_fit();
}
//*****************************************************************************
uint32 HWorlds::nbParts(uint32 worldBegin, uint32 worldEnd) const
{
    return thrust::reduce(particlesByWorld.begin()+worldBegin, particlesByWorld.begin()+worldEnd);
}
//*****************************************************************************
uint32 HWorlds::partBegin(uint32 worldBegin) const
{
    return thrust::reduce(particlesByWorld.begin(), particlesByWorld.begin()+worldBegin);
}
//*****************************************************************************
utils::HVector<uint32> HWorlds::partsBegin() const
{
    utils::HVector<uint32> result{particlesByWorld.size()};
    thrust::exclusive_scan(particlesByWorld.begin(), particlesByWorld.begin(), result.begin());
    return result;
}
//*****************************************************************************
void HWorlds::copyFromDevice(const DWorlds& device, uint32 hPartOffset, uint32 nbPart,
    uint32 dPartOffset)
{
    particles.copyFromDevice(device.particles, hPartOffset, nbPart, dPartOffset);
}
//*****************************************************************************
void HWorlds::append(const HWorlds& other)
{
    particles.append(other.particles);
    particlesByWorld.insert(particlesByWorld.end(), other.particlesByWorld.begin(), other.particlesByWorld.end());
    worldKey.insert(worldKey.end(), other.worldKey.begin(), other.worldKey.end());
    lowerLimits.insert(lowerLimits.end(), other.lowerLimits.begin(), other.lowerLimits.end());
    upperLimits.insert(upperLimits.end(), other.upperLimits.begin(), other.upperLimits.end());
    externalForces.insert(externalForces.end(), other.externalForces.begin(), other.externalForces.end());
}
//*****************************************************************************
void HWorlds::append(const World& world)
{
    particles.append(world.particles);
    particlesByWorld.push_back(world.particles.pos.size());
    worldKey.insert(worldKey.end(), world.particles.pos.size(), lowerLimits.size());
    lowerLimits.push_back(world.lowerLimits);
    upperLimits.push_back(world.upperLimits);
    externalForces.push_back(world.externalForce);
}
//*****************************************************************************
void DWorlds::reset()
{
    particles.reset();
    worldKey.reset();
    lowerLimits.reset();
    upperLimits.reset();
    externalForces.reset();
}
//*****************************************************************************
void DWorlds::reserve(uint32 nbWorld, uint32 nbPart)
{
    worldKey.reset(utils::cudaAllocator<uint32>(nbPart));
    lowerLimits.reset(utils::cudaAllocator<utils::Vec2f>(nbWorld));
    upperLimits.reset(utils::cudaAllocator<utils::Vec2f>(nbWorld));
    externalForces.reset(utils::cudaAllocator<utils::Vec2d>(nbWorld));
    particles.reserve(nbPart);
}
//*****************************************************************************
void DWorlds::copyFromHost(const HWorlds& host, uint32 hWorldOffset, uint32 nbWorld,
    uint32 hPartOffset, uint32 nbPart, uint32 dWorldOffset, uint32 dPartOffset)
{
    utils::cudaHostToDevice(host.worldKey.data()+hPartOffset, worldKey.get()+dPartOffset, nbPart);
    utils::cudaHostToDevice(host.lowerLimits.data()+hWorldOffset, lowerLimits.get()+dWorldOffset, nbWorld);
    utils::cudaHostToDevice(host.upperLimits.data()+hWorldOffset, upperLimits.get()+dWorldOffset, nbWorld);
    utils::cudaHostToDevice(host.externalForces.data()+hWorldOffset, externalForces.get()+dWorldOffset, nbWorld);
    particles.copyFromHost(host.particles, hPartOffset, nbPart, dPartOffset);
}
//*****************************************************************************
uint32 DWorlds::particleDependSize() const
{
    //1 uint32 by particle to know its world
    return sizeof(uint32);
}
//*****************************************************************************
uint32 DWorlds::worldDependSize() const
{   
    //4 floats for limits, 2 double for force
    return 4*sizeof(float)+2*sizeof(double);
}
}
