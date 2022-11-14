#include "kernelLauncher.cuh"
#include "kernels.cuh"

using namespace utils;
namespace sph {
//*****************************************************************************
NeighborsKernel::NeighborsKernel(const DWorlds& worlds, const DevicePtrGrids& grid, 
    Ptr<uint32> nbNeighbors, const SPHParameters& params, uint32 partNumber, uint32 worldBegin)
: KernelLauncher(), nbNeighbor(nbNeighbors), pos(worlds.particles.pos.get()), cellsStop(grid.cellsStop.get()),
  worldKey(worlds.worldKey.get()), gridDim(grid.dims), lowerLimits(worlds.lowerLimits.get()), 
  partNumber(partNumber), worldBegin(worldBegin), maxRadius(params.maxRadius())
{}
//*****************************************************************************
void NeighborsKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    getNeighbors<<<dimGrid, dimBlock>>>(nbNeighbor, pos, cellsStop, worldKey, gridDim,
        lowerLimits, partNumber, worldBegin, maxRadius);
}
//*****************************************************************************
GridKernel::GridKernel(const DWorlds& worlds, const DevicePtrGrids& grid, const SPHParameters& params, 
    uint32 partNumber, uint32 nbWorld, uint32 worldBegin)
: KernelLauncher(), pos(worlds.particles.pos.get()), worldKey(worlds.worldKey.get()), indexes(grid.index.get()), 
  offset(grid.offset.get()), partByCell(grid.partByCell.get()), partNumber(partNumber), nbWorld(nbWorld), gridDim(grid.dims), 
  worldBegin(worldBegin), lowerLimits(worlds.lowerLimits.get()), maxRadius(params.maxRadius())
{}
//*****************************************************************************
void GridKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);

    //Add nbWorld gridDim.x()*gridDim.y()*sizeof(uint32)
    cudaFill(partByCell, 0, nbWorld*gridDim.x()*gridDim.y());

    //compute particles cells, particle offset and number of particles per cell
    //cudaSafe("createGridKernel-before");
    createGrid<<<dimGrid, dimBlock>>>(pos, worldKey, indexes, offset, partByCell, partNumber, 
        worldBegin, gridDim, lowerLimits, maxRadius);
    //cudaSafe("createGridKernel-after");
}
//*****************************************************************************
RhoPKernel::RhoPKernel(const DWorlds& worlds, const DevicePtrGrids& grid, const SPHParameters& params, 
    uint32 partNumber, uint32 worldBegin)
: KernelLauncher(), pos(worlds.particles.pos.get()), mass(worlds.particles.mass.get()),
radius(worlds.particles.radius.get()), rho(worlds.particles.rho.get()), p(worlds.particles.p.get()),
  cellsStop(grid.cellsStop.get()), worldKey(worlds.worldKey.get()), gridDim(grid.dims), 
  lowerLimits(worlds.lowerLimits.get()), partNumber(partNumber), worldBegin(worldBegin),
  maxRadius(params.maxRadius()), rho0(params.rho0()), taitsB(params.taitsB())
{}
//*****************************************************************************
void RhoPKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    //cudaSafe("computeRhoPKernel-before");
    computeRhoP<<<dimGrid, dimBlock>>>(pos, mass, radius, rho, p, cellsStop, worldKey, gridDim, 
        lowerLimits, partNumber, worldBegin, maxRadius, rho0, taitsB);
    //cudaSafe("computeRhoPKernel-after");
}
//*****************************************************************************
ForcesKernel::ForcesKernel(const DWorlds& worlds, const DevicePtrGrids& grid, 
    const SPHParameters& params, uint32 partNumber, uint32 worldBegin, Ptr<Vec2d> fP,
    Ptr<Vec2d> fV, Ptr<Vec2d> fE)
: KernelLauncher(), pos(worlds.particles.pos.get()), mass(worlds.particles.mass.get()),
  radius(worlds.particles.radius.get()), rho(worlds.particles.rho.get()), p(worlds.particles.p.get()),
  force(worlds.particles.force.get()), vel(worlds.particles.vel.get()), externalForces(worlds.externalForces.get()),
  cellsStop(grid.cellsStop.get()), worldKey(worlds.worldKey.get()), worldBegin(worldBegin), gridDim(grid.dims),
  lowerLimits(worlds.lowerLimits.get()), partNumber(partNumber), maxRadius(params.maxRadius()),
  mu(params.mu()), fP(fP), fV(fV), fE(fE)
{}
//*****************************************************************************
void ForcesKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    //cudaSafe("computeForcesKernel-before");
    computeForces<<<dimGrid, dimBlock>>>(pos, mass, radius, rho, p, force, vel, externalForces, 
        cellsStop, worldKey, gridDim, lowerLimits, partNumber, worldBegin, 
        maxRadius, mu, fP, fV, fE);
    //cudaSafe("computeForcesKernel-after");
}
//*****************************************************************************
IntegrateKernel::IntegrateKernel(const DWorlds& worlds, float dt, uint32 partNumber)
: KernelLauncher(), pos(worlds.particles.pos.get()), rho(worlds.particles.rho.get()),
  vel(worlds.particles.vel.get()), force(worlds.particles.force.get()), dt(dt), partNumber(partNumber)
{}
//*****************************************************************************
void IntegrateKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    //cudaSafe("integrateKernel-before");
    integrate<<<dimGrid, dimBlock>>>(pos, rho, vel, force, dt, partNumber);
    //cudaSafe("integrateKernel-after");
}
//*****************************************************************************
CollisionKernel::CollisionKernel(const DWorlds& worlds, const SPHParameters& params, uint32 partNumber,
    uint32 worldBegin)
: KernelLauncher(), pos(worlds.particles.pos.get()), vel(worlds.particles.vel.get()),
  worldKey(worlds.worldKey.get()), lowerLimits(worlds.lowerLimits.get()),
  upperLimits(worlds.upperLimits.get()), elast(params.elast()), fric(params.fric()),
  partNumber(partNumber), worldBegin(worldBegin)
{}
//*****************************************************************************
void CollisionKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    //cudaSafe("collisionKernel-before");
    collision<<<dimGrid, dimBlock>>>(pos, vel, worldKey, lowerLimits, upperLimits, 
        elast, fric, partNumber, worldBegin);
    //cudaSafe("collisionKernel-after");
}
//*****************************************************************************
UpdateMassKernel::UpdateMassKernel(Ptr<float> mass, Ptr<float> radius, float avg,
    float dAvg, float dStdev, float rho0, uint32 partNumber)
: mass(mass), radius(radius), avg(avg), dAvg(dAvg), dStdev(dStdev), rho0(rho0), partNumber(partNumber)
{}
//*****************************************************************************
void UpdateMassKernel::operator()() const
{
    unsigned long maxGrid = deviceProp().maxGridSize[0];
    unsigned long maxBlock = deviceProp().maxThreadsPerBlock;
    unsigned int dimBlock = maxBlock;
    unsigned int dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);
    updateMass<<<dimGrid, dimBlock>>>(mass, radius, avg, dAvg, dStdev, rho0, partNumber);
}
}
