#include "kernelLauncher.cuh"
#include "kernels.cuh"

using namespace utils;

namespace film {
//*****************************************************************************
InitKernel::InitKernel(Ptr<float> intensities, Ptr<Vec2f> centers, Ptr<Vec2f> lowerLimits,
    Ptr<Vec2f> upperLimits, Ptr<Vec2f> pos, Ptr<float> mass, Ptr<float> radius,
    Ptr<uint32> worldKey, ConstPtr<Vec2f> possiblePos, ConstPtr<uint32> partStop,
    Ptr<curandState> states, Ptr<bool> mark, float intensityCoeff, float avgMass,
    float stdevMass, float rho0, const Vec2<uint32>& dim, uint32 partMax)
: intensities(intensities), centers(centers), lowerLimits(lowerLimits), upperLimits(upperLimits),
  pos(pos), mass(mass), radius(radius), worldKey(worldKey), possiblePos(possiblePos),
  partStop(partStop), states(states), mark(mark), intensityCoeff(intensityCoeff),
  avgMass(avgMass), stdevMass(stdevMass), rho0(rho0), dim(dim), partMax(partMax)
{}
//*****************************************************************************
void InitKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((dim.x()*dim.y()+dimBlock-1u)/dimBlock,maxGrid);
    initSPH<<<dimGrid, dimBlock>>>(intensities, centers, lowerLimits, upperLimits,
        pos, mass, radius, worldKey, possiblePos, partStop, states, mark, intensityCoeff,
        avgMass, stdevMass, rho0, dim, partMax);
}
//*****************************************************************************
RandomPosKernel::RandomPosKernel(Ptr<float> intensities, Ptr<Vec2f> centers,
    Ptr<Vec2f> lowerLimits, Ptr<Vec2f> upperLimits, Ptr<curandState> states,
    Ptr<bool> mark, Ptr<uint32> nbPart, float intensityCoeff, const Vec2<uint32>& dim, uint32 maxPart)
: intensities(intensities), centers(centers), lowerLimits(lowerLimits), upperLimits(upperLimits),
  states(states), mark(mark), nbPart(nbPart), intensityCoeff(intensityCoeff), dim(dim), maxPart(maxPart)
{}
//*****************************************************************************
void RandomPosKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((dim.x()*dim.y()+dimBlock-1u)/dimBlock,maxGrid);
    initRandomPos<<<dimGrid, dimBlock>>>(intensities, centers, lowerLimits, upperLimits,
        states, mark, nbPart, intensityCoeff, dim, maxPart);
}
//*****************************************************************************
InitRandomKernel::InitRandomKernel(Ptr<Vec2f> pos, Ptr<float> mass, Ptr<float> radius,
    Ptr<uint32> worldKey, ConstPtr<Vec2f> possiblePos, ConstPtr<uint32> partStop,
    Ptr<curandState> states, ConstPtr<bool> mark, float avgMass, float stdevMass,
    float rho0, const Vec2<uint32>& dim, uint32 maxPart)
: pos(pos), mass(mass), radius(radius), worldKey(worldKey), possiblePos(possiblePos),
  partStop(partStop), states(states), mark(mark), avgMass(avgMass), stdevMass(stdevMass),
  rho0(rho0), dim(dim), maxPart(maxPart)
{}
//*****************************************************************************
void InitRandomKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((dim.x()*dim.y()+dimBlock-1u)/dimBlock,maxGrid);
    initRandomSPH<<<dimGrid, dimBlock>>>(pos, mass, radius, worldKey, possiblePos, partStop,
        states, mark, avgMass, stdevMass, rho0, dim, maxPart);
}
//*****************************************************************************
GridKernel::GridKernel(ConstPtr<Vec2f> pos, Ptr<uint32> indexes, Ptr<uint32> offset,
    Ptr<uint32> partByCell, uint32 partNumber, const Vec2<uint32>& gridDim,
    const Vec2f& pixelSize, const thrust::pair<uint32,uint32>& firstPixel)
: KernelLauncher(), pos(pos), indexes(indexes), offset(offset), partByCell(partByCell),
  partNumber(partNumber), gridDim(gridDim), pixelSize(pixelSize), firstPixel(firstPixel)
{}
//*****************************************************************************
void GridKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((partNumber+dimBlock-1u)/dimBlock,maxGrid);

    //Add nbWorld gridDim.x()*gridDim.y()*sizeof(uint32)
    cudaFill(partByCell, 0, gridDim.x()*gridDim.y());

    //compute particles cells, particle offset and number of particles per cell
    //cudaSafe("createGridKernel-before");
    createGrid<<<dimGrid, dimBlock>>>(pos, indexes, offset, partByCell,
        partNumber, gridDim, pixelSize, firstPixel);
    //cudaSafe("createGridKernel-after");
}
//*****************************************************************************
DensityKernel::DensityKernel(Ptr<float> densities, ConstPtr<Vec2f> pos,
    ConstPtr<float> grainsRadius, ConstPtr<uint32> cellsStop, const thrust::pair<uint32, uint32>& nbSamples,
    float maxGrainRadius, uint32 width, uint32 height, const Vec2f& ratio,
    const thrust::pair<uint32,uint32>& firstPixel, bool debug)
: KernelLauncher(), densities(densities), pos(pos), grainsRadius(grainsRadius),
  cellsStop(cellsStop), nbSamples(nbSamples), maxGrainRadius(maxGrainRadius), width(width),
  height(height), ratio(ratio), firstPixel(firstPixel), debug(debug)
{}
//*****************************************************************************
void DensityKernel::operator()() const
{
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    dim3 blocks;
    dim3 thrds;
    thrds.x = cbrt(maxBlock);
    thrds.y = cbrt(maxBlock);
    thrds.z = cbrt(maxBlock);
    blocks.x = umin((nbSamples.first*nbSamples.second + maxBlock-1) / cbrt(maxBlock),deviceProp().maxGridSize[0]);
    blocks.y = umin((width + maxBlock-1) / cbrt(maxBlock), deviceProp().maxGridSize[1]);
    blocks.z = umin((height + maxBlock - 1) / cbrt(maxBlock), deviceProp().maxGridSize[2]);
    computeDensity<<<blocks, thrds>>>(densities, pos, grainsRadius, cellsStop, nbSamples,
        maxGrainRadius, width, height, ratio, firstPixel, debug);
}
//*****************************************************************************
RandomKernel::RandomKernel(const Ptr<curandState> states, uint64 seed, uint32 nbStates)
: KernelLauncher(), states(states), seed(seed), nbStates(nbStates)
{}
//*****************************************************************************
void RandomKernel::operator()() const
{
    uint64 maxGrid = deviceProp().maxGridSize[0];
    uint64 maxBlock = deviceProp().maxThreadsPerBlock;
    uint32 dimBlock = maxBlock;
    uint32 dimGrid = umin((nbStates+dimBlock-1u)/dimBlock,maxGrid);

    initRandomState<<<dimGrid, dimBlock>>>(states, seed, nbStates);
}
}
