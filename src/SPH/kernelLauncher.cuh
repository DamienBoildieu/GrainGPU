#pragma once
#include "utils/cuda.cuh"
#include "utils/memory.cuh"
#include "grid.cuh"
#include "world.cuh"
#include "sphParameters.cuh"

namespace sph {
//*****************************************************************************
class NeighborsKernel : public utils::KernelLauncher
{
public:
    NeighborsKernel(const DWorlds& worlds, const DevicePtrGrids& grid, utils::Ptr<uint32> nbNeighbors,
        const SPHParameters& params, uint32 partNumber, uint32 worldBegin);
    void operator()() const override;
private:
    utils::Ptr<uint32> nbNeighbor;
    utils::ConstPtr<utils::Vec2f> pos;
    utils::ConstPtr<uint32> cellsStop;
    utils::ConstPtr<uint32> worldKey;
    utils::Vec2<uint32> gridDim;
    utils::ConstPtr<utils::Vec2f> lowerLimits;
    uint32 partNumber;
    uint32 worldBegin;
    float maxRadius;
};
//*****************************************************************************
class GridKernel : public utils::KernelLauncher
{
public:
    GridKernel(const DWorlds& worlds, const DevicePtrGrids& grid, const SPHParameters& params, 
        uint32 partNumber, uint32 nbWorld, uint32 worldBegin);
    void operator()() const override;
private:
    utils::ConstPtr<utils::Vec2f> pos;
    utils::ConstPtr<uint32> worldKey;
    utils::Ptr<uint32> indexes;
    utils::Ptr<uint32> offset;
    utils::Ptr<uint32> partByCell;
    uint32 partNumber;
    uint32 nbWorld;
    uint32 worldBegin;
    utils::Vec2<uint32> gridDim;
    utils::ConstPtr<utils::Vec2f> lowerLimits;
    float maxRadius;
};
//*****************************************************************************
class RhoPKernel : public utils::KernelLauncher
{
public:
    RhoPKernel(const DWorlds& worlds, const DevicePtrGrids& grid, const SPHParameters& params, 
        uint32 partNumber, uint32 worldBegin);
    void operator()() const override;
private:
    utils::ConstPtr<utils::Vec2f> pos;
    utils::ConstPtr<float> mass;
    utils::ConstPtr<float> radius;
    utils::Ptr<float> rho;
    utils::Ptr<float> p;
    utils::ConstPtr<uint32> cellsStop;
    utils::ConstPtr<uint32> worldKey;
    utils::Vec2<uint32> gridDim;
    utils::ConstPtr<utils::Vec2f> lowerLimits;
    uint32 partNumber;
    uint32 worldBegin;
    float maxRadius;
    float rho0;
    float taitsB;
};
//*****************************************************************************
class ForcesKernel : public utils::KernelLauncher
{
public:
    ForcesKernel(const DWorlds& worlds, const DevicePtrGrids& grid, const SPHParameters& params,
        uint32 partNumber, uint32 worldBegin, utils::Ptr<utils::Vec2d> fP=nullptr, utils::Ptr<utils::Vec2d> fV=nullptr,
        utils::Ptr<utils::Vec2d> fE=nullptr);
    void operator()() const override;
private:
    utils::ConstPtr<utils::Vec2f> pos;
    utils::ConstPtr<float> mass;
    utils::ConstPtr<float> radius;
    utils::ConstPtr<float> rho;
    utils::ConstPtr<float> p;
    utils::Ptr<utils::Vec2d> force;
    utils::ConstPtr<utils::Vec2f> vel;
    utils::ConstPtr<utils::Vec2d> externalForces;
    utils::Ptr<uint32> cellsStop;
    utils::ConstPtr<uint32> worldKey;
    utils::Vec2<uint32> gridDim;
    utils::ConstPtr<utils::Vec2f> lowerLimits;
    uint32 partNumber;
    uint32 worldBegin;
    float maxRadius;
    float mu;
    utils::Ptr<utils::Vec2d> fP;
    utils::Ptr<utils::Vec2d> fV;
    utils::Ptr<utils::Vec2d> fE;
};
//*****************************************************************************
class IntegrateKernel : public utils::KernelLauncher
{
public:
    IntegrateKernel(const DWorlds& worlds, float dt, uint32 partNumber);
    void operator()() const override;
private:
    utils::Ptr<utils::Vec2f> pos;
    utils::ConstPtr<float> rho;
    utils::Ptr<utils::Vec2f> vel;
    utils::ConstPtr<utils::Vec2d> force;
    float dt;
    uint32 partNumber;
};
//*****************************************************************************
class CollisionKernel : public utils::KernelLauncher
{
public:
    CollisionKernel(const DWorlds& worlds, const SPHParameters& params, uint32 partNumber,
        uint32 worldBegin);
    void operator()() const override;
private:
    utils::Ptr<utils::Vec2f> pos;
    utils::Ptr<utils::Vec2f> vel;
    utils::ConstPtr<uint32> worldKey;
    utils::ConstPtr<utils::Vec2f> lowerLimits;
    utils::ConstPtr<utils::Vec2f> upperLimits;
    float elast;
    float fric;
    uint32 partNumber;
    uint32 worldBegin;
};
//*****************************************************************************
class UpdateMassKernel : public utils::KernelLauncher
{
public:
    UpdateMassKernel(utils::Ptr<float> mass, utils::Ptr<float> radius,
        float avg, float dAvg, float dStdev, float rho0, uint32 partNumber);
    void operator()() const override;
private:
    utils::Ptr<float> mass;
    utils::Ptr<float> radius;
    float avg;
    float dAvg;
    float dStdev;
    float rho0;
    uint32 partNumber;
};
}
