#include "sph.cuh"
#include <cmath>
#include <thrust/scan.h>
#include "utils/cuda.cuh"
#include "kernelLauncher.cuh"
#include <iostream>
#include <fstream>
#include "utils/define.h"
#include "utils/stats.cuh"
#include "utils/system.h"
#include "utils/functors.cuh"

using namespace utils;

namespace sph {
//*****************************************************************************
SPH::SPH(const SPHParameters& params, bool writeResults, bool writeForceStats)
: mParams(params), mWriteResults(writeResults), mWriteForceStats(false),
  mNbIte(0u)
{
    HANDLE_ERROR(cudaGetDeviceProperties(&mProps, chooseBestDevice()));
    if (mWriteResults || mWriteForceStats)
        createDirs();
}
//*****************************************************************************
SPH::SPH(const SPH& other)
: mHostWorlds(other.mHostWorlds), mDeviceWorlds(), mDeviceGrid(), mParams(other.mParams),
  mProps(other.mProps), mAllocatedParticles(0), mAllocatedWorlds(0), mWorldBegin(0),
  mWriteResults(other.mWriteResults), mWriteForceStats(other.mWriteForceStats), mNbIte(other.mNbIte)
{
    mDeviceGrid.dims = other.mDeviceGrid.dims;
    if (mWriteResults || mWriteForceStats)
        createDirs();
}
//*****************************************************************************
SPH& SPH::operator=(const SPH& right)
{
    if (&right!=this) {
        mHostWorlds = right.mHostWorlds;
        mDeviceWorlds = {};
        mDeviceGrid = {};
        mDeviceGrid.dims = right.mDeviceGrid.dims;
        mParams = right.mParams;
        mProps = right.mProps;
        mAllocatedParticles = 0;
        mAllocatedWorlds = 0;
        mWorldBegin = 0;
        mWriteResults = right.mWriteResults;
        mWriteForceStats = right.mWriteForceStats;
        mNbIte = right.mNbIte;
    }
    return *this;
}
//*****************************************************************************
void SPH::addForce(uint32 world, const Vec2d& force)
{
    assert(world<mHostWorlds.externalForces.size());
    mHostWorlds.externalForces[world] += force;
}
//*****************************************************************************
void SPH::setForce(uint32 world, const Vec2d& force)
{
    assert(world<mHostWorlds.externalForces.size());
    mHostWorlds.externalForces[world] = force;
}
//*****************************************************************************
void SPH::resetForce(uint32 world)
{
    assert(world<mHostWorlds.externalForces.size());
    mHostWorlds.externalForces[world] = Vec2d(0.);
}
//*****************************************************************************
void SPH::update(float dt)
{
    if (mAllocatedParticles==0)
        return;
    if (mWriteResults || mWriteForceStats) {
        createDirs();
    }
    std::cerr << "allocated worlds : " << mAllocatedWorlds << " -- allocated particles : " << mAllocatedParticles << std::endl;
    computeGrid();
    if (mWriteResults) {
        DUniqueArrayPtr<uint32> nbNeighs{cudaAllocator<uint32>(mAllocatedParticles)};
        NeighborsKernel{mDeviceWorlds, mDeviceGrid, nbNeighs.get(), mParams, mAllocatedParticles,
            mWorldBegin}();

        writeNbNeighs(nbNeighs.get());
        writeBeforePos();
    }
    
    RhoPKernel{mDeviceWorlds, mDeviceGrid, mParams, mAllocatedParticles, mWorldBegin}();

    if (mWriteForceStats) {
        DUniqueArrayPtr<Vec2d> fP{cudaAllocator<Vec2d>(mAllocatedParticles)};
        DUniqueArrayPtr<Vec2d> fV{cudaAllocator<Vec2d>(mAllocatedParticles)};
        DUniqueArrayPtr<Vec2d> fE{cudaAllocator<Vec2d>(mAllocatedParticles)};
        ForcesKernel{mDeviceWorlds, mDeviceGrid, mParams, mAllocatedParticles, mWorldBegin, fP.get(), fV.get(), fE.get()}();
        writeForceStats(fP.get(), fV.get(), fE.get());
    } else {
        ForcesKernel{mDeviceWorlds, mDeviceGrid, mParams, mAllocatedParticles, mWorldBegin}();
    }
    
    IntegrateKernel{mDeviceWorlds, dt, mAllocatedParticles}();

    CollisionKernel{mDeviceWorlds, mParams, mAllocatedParticles, mWorldBegin}();

    if (mWriteResults)
        writeResults();
    mNbIte++;
}
//*****************************************************************************
thrust::pair<uint32,uint32> SPH::hostToDevice(uint32 begin, uint32 end)
{
    if (mAllocatedWorlds == 0u)
        mWorldBegin = begin;
    else if (mWorldBegin > begin)
        mWorldBegin = begin;
    const uint32 nbParts = mHostWorlds.nbParts(begin, end);
    const uint32 nbWorlds = end-begin;
    thrust::pair<uint32,uint32> dOffset{mAllocatedWorlds, mAllocatedParticles};
    mDeviceWorlds.copyFromHost(mHostWorlds, begin, nbWorlds, mHostWorlds.partBegin(begin),
        nbParts, mAllocatedWorlds, mAllocatedParticles);
    mAllocatedWorlds += nbWorlds;
    mAllocatedParticles += nbParts;
    return dOffset;
}
//*****************************************************************************
void SPH::deviceToHost(uint32 begin, uint32 end, const thrust::pair<uint32,uint32>& dOffset)
{
    const uint32 nbParts = mHostWorlds.nbParts(begin, end);
    mHostWorlds.copyFromDevice(mDeviceWorlds, mHostWorlds.partBegin(begin), nbParts, dOffset.second);
}
//*****************************************************************************
void SPH::reserveDevice(uint32 nbWorld, uint32 nbPart)
{
    mAllocatedParticles = 0;
    mAllocatedWorlds = 0;
    mWorldBegin = 0;
    mDeviceWorlds.reserve(nbWorld, nbPart);
    mDeviceGrid.reserve(nbWorld, nbPart);
}
//*****************************************************************************
void SPH::cleanDevice()
{
    mDeviceWorlds.reset();
    mDeviceGrid.reset();
    mAllocatedParticles = 0;
    mAllocatedWorlds = 0;
    mWorldBegin = 0;
}
//*****************************************************************************
void SPH::cleanHost()
{
    mDeviceWorlds.reset();
}
//*****************************************************************************
uint32 SPH::nbWorlds()
{
    return mHostWorlds.particlesByWorld.size();
}
//*****************************************************************************
void SPH::addWorld(const World& world)
{
    mHostWorlds.append(world);
}
//*****************************************************************************
const DWorlds& SPH::deviceWorlds() const
{
    return mDeviceWorlds;
}
//*****************************************************************************
DWorlds& SPH::deviceWorlds()
{
    return mDeviceWorlds;
}
//*****************************************************************************
const HWorlds& SPH::hostWorlds() const
{
    return mHostWorlds;
}
//*****************************************************************************
HWorlds& SPH::hostWorlds()
{
    return mHostWorlds;
}
//*****************************************************************************
void SPH::setGridDim(const utils::Vec2<uint32>& dims)
{
    mDeviceGrid.dims = dims;
}
//*****************************************************************************
const Vec2f& SPH::lowerLimits(uint32 world) const
{
    assert(world<mHostWorlds.lowerLimits.size());
    return mHostWorlds.lowerLimits[world];
}
//*****************************************************************************
void SPH::setLowerLimits(uint32 world, const Vec2f& lower)
{
    assert(world<mHostWorlds.lowerLimits.size());
    mHostWorlds.lowerLimits[world] = lower;
}
//*****************************************************************************
const Vec2f& SPH::upperLimits(uint32 world) const
{
    assert(world<mHostWorlds.upperLimits.size());
    return mHostWorlds.upperLimits[world];
}
//*****************************************************************************
void SPH::setUpperLimits(uint32 world, const Vec2f& upper)
{
    assert(world<mHostWorlds.upperLimits.size());
    mHostWorlds.upperLimits[world] = upper;
}
//*****************************************************************************
const SPHParameters& SPH::params() const
{
    return mParams;
}
//*****************************************************************************
void SPH::setAvgMass(float mass)
{
    const float oldAvgMass = mParams.avgMass();
    const float dAvgMass = mass - oldAvgMass;
    const uint32 nbPart = mHostWorlds.particles.mass.size();
    DUniqueArrayPtr<float> dMass{cudaAllocator<float>(nbPart)};
    cudaHostToDevice(mHostWorlds.particles.mass.data(), dMass.get(), nbPart);
    DUniqueArrayPtr<float> dRadius{cudaAllocator<float>(nbPart)};

    UpdateMassKernel{dMass.get(), dRadius.get(), oldAvgMass, dAvgMass, 1.f, mParams.rho0(), nbPart}();
    cudaDeviceToHost(dMass.get(), mHostWorlds.particles.mass.data(), nbPart);
    cudaDeviceToHost(dRadius.get(), mHostWorlds.particles.radius.data(), nbPart);
    mParams.setAvgMass(mass);
}
//*****************************************************************************
void SPH::setMassStdev(float massStdev)
{
    assert(massStdev!=0.);
    const float oldMassStdev = mParams.massStdev();
    float dMassStdev = massStdev/oldMassStdev;
    if (isinf(dMassStdev))
        dMassStdev = 1.;
    const uint32 nbPart = mHostWorlds.particles.mass.size();
    DUniqueArrayPtr<float> dMass{cudaAllocator<float>(nbPart)};
    cudaHostToDevice(mHostWorlds.particles.mass.data(), dMass.get(), nbPart);
    DUniqueArrayPtr<float> dRadius{cudaAllocator<float>(nbPart)};

    UpdateMassKernel{dMass.get(), dRadius.get(), mParams.avgMass(), 0.f, dMassStdev,
        mParams.rho0(), nbPart}();
    cudaDeviceToHost(dMass.get(), mHostWorlds.particles.mass.data(), nbPart);
    cudaDeviceToHost(dRadius.get(), mHostWorlds.particles.radius.data(), nbPart);
    mParams.setMassStdev(massStdev);
}
//*****************************************************************************
void SPH::setRho0(float rho0)
{
    const uint32 nbPart = mHostWorlds.particles.mass.size();
    DUniqueArrayPtr<float> dMass{cudaAllocator<float>(nbPart)};
    cudaHostToDevice(mHostWorlds.particles.mass.data(), dMass.get(), nbPart);
    DUniqueArrayPtr<float> dRadius{cudaAllocator<float>(nbPart)};
    DPtr<float> tMass{dMass.get()};
    DPtr<float> tRadius{dRadius.get()};
    transform(tMass, tMass+nbPart, tRadius,
        [=] DEVICE (float val) { return radius(val, rho0);});
    cudaDeviceToHost(dRadius.get(), mHostWorlds.particles.radius.data(), nbPart);
    mParams.setRho0(rho0);
}
//*****************************************************************************
void SPH::setMu(float mu)
{
    mParams.setMu(mu);
}
//*****************************************************************************
void SPH::setElast(float elast)
{
    mParams.setElast(elast);
}
//*****************************************************************************
void SPH::setFric(float fric)
{
    mParams.setFric(fric);
}
//*****************************************************************************
void SPH::setParams(const SPHParameters& parameters)
{
    mParams = parameters;
}
//*****************************************************************************
void SPH::setWriteResults(bool writeResults)
{
    mWriteResults = writeResults;
}
//*****************************************************************************
void SPH::setWriteForceStats(bool writeForceStats)
{
    mWriteForceStats = writeForceStats;
}
//*****************************************************************************
const cudaDeviceProp& SPH::cudaProps() const
{
    return mProps;
}
//*****************************************************************************
uint32 SPH::allocatedParticles() const
{
    return mAllocatedParticles;
}
//*****************************************************************************
void SPH::setWorldBegin(uint32 worldBegin)
{
    mWorldBegin = worldBegin;
}
//*****************************************************************************
uint32 SPH::allocableWorld(uint32 availableSize, uint32 startPos) const
{
    constexpr uint32 particleSize = sizeof(Particle);
    const uint32 gridSize = mDeviceGrid.dimDependSize()*mDeviceGrid.dims.x()*mDeviceGrid.dims.y();
    const uint32 gridDimSize = mDeviceGrid.constSize();
    //we let space for params and ignore particles number because we don't use all parameters at the same time
    constexpr uint32 paramsSize = sizeof(SPHParameters);
    const uint32 worldSize = mDeviceWorlds.worldDependSize();
    const uint32 constNumerator = availableSize-gridDimSize-paramsSize;
    const uint32 denominator = particleSize+mDeviceGrid.particleDependSize()+mDeviceWorlds.particleDependSize()+3.*sizeof(Vec2d);
    
    uint32 maxParticle = 0;
    uint32 nbWorld = 0;
    uint32 nbParticle = 0;
    auto nbPartWorld = mHostWorlds.particlesByWorld.begin()+startPos;
    do {
        nbParticle += *(nbPartWorld++);
        nbWorld++;
        const uint32 numerator = constNumerator-nbWorld*gridSize*worldSize;
        maxParticle = numerator/denominator;
    } while (nbParticle < maxParticle && nbPartWorld!=mHostWorlds.particlesByWorld.end());
    return nbWorld;
}
//*****************************************************************************
void SPH::computeGrid()
{
    std::cerr << "computeGrid " << mAllocatedWorlds*mDeviceGrid.dims.x()*mDeviceGrid.dims.y() << std::endl;
    const uint32 worldEnd = mAllocatedWorlds*mDeviceGrid.dims.x()*mDeviceGrid.dims.y();
    GridKernel{mDeviceWorlds, mDeviceGrid, mParams, mAllocatedParticles, mAllocatedWorlds, mWorldBegin}();
    //compute the last index in particle list for each grid cell
    DPtr<uint32> partByCell{mDeviceGrid.partByCell.get()};
    DPtr<uint32> cellsStop{mDeviceGrid.cellsStop.get()};
    inclusive_scan(partByCell, partByCell+worldEnd, cellsStop);
    //compute the new particles positions
    OffsetFunctor functor{mDeviceGrid.cellsStop.get()};
    DPtr<uint32> index{mDeviceGrid.index.get()};
    DPtr<uint32> offset{mDeviceGrid.offset.get()};
    transform(index, index+mAllocatedParticles, offset, offset, functor);
    //sort particles
    DPtr<Vec2f> vel{mDeviceWorlds.particles.vel.get()};
    DUniqueArrayPtr<Vec2f> velResult{cudaAllocator<Vec2f>(mAllocatedParticles)};
    DPtr<Vec2f> tVelResult{velResult.get()};
    scatter(vel, vel+mAllocatedParticles, offset, tVelResult);
    mDeviceWorlds.particles.vel = std::move(velResult);
    DPtr<Vec2f> pos{mDeviceWorlds.particles.pos.get()};
    DUniqueArrayPtr<Vec2f> posResult{cudaAllocator<Vec2f>(mAllocatedParticles)};
    DPtr<Vec2f> tPosResult{posResult.get()};
    scatter(pos, pos+mAllocatedParticles, offset, tPosResult);
    mDeviceWorlds.particles.pos = std::move(posResult);
}
//*****************************************************************************
void SPH::createDirs() const
{
    createDir(logDir);
    createDir(forceDir.c_str());
    if (mWriteResults) {
        createDir(rhoDir.c_str());
        createDir(pressDir.c_str());
        createDir(velDir.c_str());
        createDir(beforePosDir.c_str());
        createDir(posDir.c_str());
        createDir(neighDir.c_str());
    }
}
//*****************************************************************************
void SPH::writeBeforePos() const
{
    std::string iteFile{std::string{dirSep}+std::string{"ite"}+std::to_string(mNbIte)+std::string{".txt"}};
    writeDeviceArray(beforePosDir+iteFile, mDeviceWorlds.particles.pos.get(), mAllocatedParticles);
}
//*****************************************************************************
void SPH::writeNbNeighs(utils::ConstPtr<uint32> nbNeighs) const
{
    std::string iteFile{std::string{dirSep}+std::string{"ite"}+std::to_string(mNbIte)+std::string{".txt"}};
    writeDeviceArray(neighDir+iteFile, nbNeighs, mAllocatedParticles);
}
//*****************************************************************************
void SPH::writeResults() const
{
    std::string iteFile{std::string{dirSep}+std::string{"ite"}+std::to_string(mNbIte)+std::string{".txt"}};
    writeDeviceArray(rhoDir+iteFile, mDeviceWorlds.particles.rho.get(), mAllocatedParticles);
    writeDeviceArray(pressDir+iteFile, mDeviceWorlds.particles.p.get(), mAllocatedParticles);
    writeDeviceArray(forceDir+iteFile, mDeviceWorlds.particles.force.get(), mAllocatedParticles);
    writeDeviceArray(velDir+iteFile, mDeviceWorlds.particles.vel.get(), mAllocatedParticles);
    writeDeviceArray(posDir+iteFile, mDeviceWorlds.particles.pos.get(), mAllocatedParticles);
}
//*****************************************************************************
void SPH::writeForceStats(ConstPtr<Vec2d> fP, ConstPtr<Vec2d> fV, ConstPtr<Vec2d> fE) const
{
    DUniqueArrayPtr<uint32> outWorldKey{cudaAllocator<uint32>(mAllocatedWorlds)};
    DVector<uint32> tNbParticles(mHostWorlds.particlesByWorld.begin()+mWorldBegin,
                                        mHostWorlds.particlesByWorld.begin()+mWorldBegin+mAllocatedWorlds);

    DUniqueArrayPtr<Vec2d> fPMin = mins(fP, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fPMax = maxs(fP, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fPAvg = avgs(fP, mDeviceWorlds.worldKey.get(), tNbParticles.data().get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);

    DUniqueArrayPtr<Vec2d> fVMin = mins(fV, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fVMax = maxs(fV, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fVAvg = avgs(fV, mDeviceWorlds.worldKey.get(), tNbParticles.data().get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);

    DUniqueArrayPtr<Vec2d> fEMin = mins(fE, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fEMax = maxs(fE, mDeviceWorlds.worldKey.get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);
    DUniqueArrayPtr<Vec2d> fEAvg = avgs(fE, mDeviceWorlds.worldKey.get(), tNbParticles.data().get(), mAllocatedParticles, outWorldKey.get(), mAllocatedWorlds);

    std::ofstream nbPartFile{std::string{logDir}+std::string{dirSep}+std::string{"nbPart.txt"}};
    HVector<uint32> nbParticles(tNbParticles);
    writeArray(nbPartFile, nbParticles);
    std::ofstream configFile{std::string{logDir}+std::string{dirSep}+std::string{"config.txt"}};
    configFile << "Average mass : " << mParams.avgMass() << std::endl << " average Radius : " << mParams.avgRadius() << std::endl
               << "Rest density : " << mParams.rho0() << std::endl << "Mu : " << mParams.mu() << std::endl
               << "Elasticity coeff : " << mParams.elast() << std::endl << "Friction coeff : " << mParams.fric() << std::endl;

    std::string iteFile{std::string{"ite"}+std::to_string(mNbIte)+std::string{".txt"}};
    writeDeviceArray(forceDir+std::string{dirSep}+"fPMin"+iteFile, fPMin.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fPMax"+iteFile, fPMax.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fPAvg"+iteFile, fPAvg.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fVMin"+iteFile, fVMin.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fVMax"+iteFile, fVMax.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fVAvg"+iteFile, fVAvg.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fEMin"+iteFile, fEMin.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fEMax"+iteFile, fEMax.get(), mAllocatedWorlds);
    writeDeviceArray(forceDir+std::string{dirSep}+"fEAvg"+iteFile, fEAvg.get(), mAllocatedWorlds);    
}
}
