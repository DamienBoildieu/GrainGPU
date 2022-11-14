#pragma once
#include "sphParameters.cuh"
#include "particles.cuh"
#include "grid.cuh"
#include "utils/vec.cuh"
#include "world.cuh"

namespace sph {
class SPH {
public:
    SPH(const SPHParameters& params, bool writeResults=false, bool writeForceStats=false);
    SPH(const SPH& other);
    ~SPH() = default;
    SPH& operator=(const SPH& right);

    void addForce(uint32 world, const utils::Vec2d& force);
    void setForce(uint32 world, const utils::Vec2d& force);
    void resetForce(uint32 world);

    void update(float dt);

    //return a pair with the pixels and particles offset on device list
    thrust::pair<uint32,uint32> hostToDevice(uint32 begin, uint32 end);
    void deviceToHost(uint32 begin, uint32 end, const thrust::pair<uint32,uint32>& dOffset={});
    void reserveDevice(uint32 nbWorld, uint32 nbPart);
    void cleanDevice();
    void cleanHost();

    uint32 nbWorlds();

    void addWorld(const World& world);

    const DWorlds& deviceWorlds() const;
    DWorlds& deviceWorlds();
    const HWorlds& hostWorlds() const;
    HWorlds& hostWorlds();

    void setGridDim(const utils::Vec2<uint32>& dims);

    const utils::Vec2f& lowerLimits(uint32 world) const;
    void setLowerLimits(uint32 world, const utils::Vec2f& lower);
    const utils::Vec2f& upperLimits(uint32 world) const;
    void setUpperLimits(uint32 world, const utils::Vec2f& upper);

    const SPHParameters& params() const;
    void setAvgMass(float mass);
    void setMassStdev(float massStdev);
    void setRho0(float rho0);
    void setMu(float mu);
    void setElast(float elast);
    void setFric(float fric);
    void setParams(const SPHParameters& parameters);
    void setWriteResults(bool writeResults);
    void setWriteForceStats(bool forceStats);
    const cudaDeviceProp& cudaProps() const;

    uint32 allocatedParticles() const;
    void setWorldBegin(uint32 worldBegin);
    
    uint32 allocableWorld(uint32 availableSize, uint32 startPos) const;
    
private:
    void computeGrid();
    void createDirs() const;
    void writeBeforePos() const;
    void writeNbNeighs(utils::ConstPtr<uint32> neighs) const;
    void writeResults() const;
    void writeForceStats(utils::ConstPtr<utils::Vec2d> fP, utils::ConstPtr<utils::Vec2d> fV,
        utils::ConstPtr<utils::Vec2d> fE) const;
    HWorlds mHostWorlds;
    DWorlds mDeviceWorlds;
    DevicePtrGrids mDeviceGrid;

    uint32 mAllocatedParticles;
    uint32 mAllocatedWorlds;
    uint32 mWorldBegin;
    
    SPHParameters mParams;
    cudaDeviceProp mProps;

    bool mWriteResults;
    bool mWriteForceStats;
    uint32 mNbIte;
};
}
