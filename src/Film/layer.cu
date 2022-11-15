#include "layer.cuh"
#include <opencv2/opencv.hpp>

using namespace sph;
using namespace utils;

namespace film {
//*****************************************************************************
Layer::Layer(const Layer& other)
: mProps(other.mProps), mSph(other.mSph), mHostPixels(other.mHostPixels), mWidth(other.mWidth),
  mHeight(other.mHeight), mIntensityCoeff(other.mIntensityCoeff), mTx(other.mTx), mTy(other.mTy)
{}
//*****************************************************************************
Layer& Layer::operator=(const Layer& right)
{
    if (this!=&right) {
        mProps = right.mProps;
        mSph = right.mSph;
        mHostPixels = right.mHostPixels;
        mWidth = right.mWidth;
        mHeight = right.mHeight;
        mIntensityCoeff = right.mIntensityCoeff;
        mTx = right.mTx;
        mTy = right.mTy;
    }
    return *this;
}
//*****************************************************************************
const HostVectorPixels& Layer::hostPixels() const
{
    return mHostPixels;
}
//*****************************************************************************
uint32 Layer::width() const
{
    return mWidth;
}
//*****************************************************************************
uint32 Layer::height() const
{
    return mHeight;
}
//*****************************************************************************
void Layer::setAvgMass(float avgMass)
{
    mSph.setAvgMass(avgMass);
}
//*****************************************************************************
void Layer::setMassStdev(float massStdev)
{
    mSph.setMassStdev(massStdev);
}
//*****************************************************************************
void Layer::setRho0(float rho0)
{
    mSph.setRho0(rho0);
}
//*****************************************************************************
void Layer::setMu(float mu)
{
    mSph.setMu(mu);
}
//*****************************************************************************
void Layer::setElast(float elast)
{
    mSph.setElast(elast);
}
//*****************************************************************************
void Layer::setFric(float fric)
{
    mSph.setFric(fric);
}
//*****************************************************************************
void Layer::setSimuParams(const sph::SPHParameters& parameters)
{
    mSph.setParams(parameters);
}
//*****************************************************************************
void Layer::setWriteResults(bool writeResults)
{
    mSph.setWriteResults(writeResults);
}
//*****************************************************************************
void Layer::setWriteForceStats(bool forceStats)
{
    mSph.setWriteForceStats(forceStats);
}
//*****************************************************************************
void Layer::update(float dt, uint32 iterations)
{
    uint32 startPixel = 0u;   
    do {
        const uint32 nbPixels = mSph.allocableWorld(mProps.totalGlobalMem, startPixel);
        const uint32 nbPart = mSph.hostWorlds().nbParts(startPixel, startPixel+nbPixels);
        mSph.reserveDevice(nbPixels, nbPart);
        mSph.hostToDevice(startPixel, startPixel+nbPixels);
        for (uint32 i=0; i<iterations; i++) {
            mSph.update(dt);
        }
        mSph.deviceToHost(startPixel, startPixel+nbPixels);
        startPixel+= nbPixels;
    } while(startPixel<mHeight*mWidth);
    mSph.cleanDevice();
}
//*****************************************************************************
void Layer::update(const thrust::pair<uint32,uint32>& startPixel, const thrust::pair<uint32,uint32>& nbPixels,
    float dt, uint32 iterations)
{
    uint32 firstPixel = startPixel.second*mWidth+startPixel.first;
    //count number of particles
    uint32 totalNbPart = 0u;
    for (uint32 stripBegin = firstPixel; stripBegin<(startPixel.second+nbPixels.second)*mWidth; stripBegin+=mWidth) {
        totalNbPart += mSph.hostWorlds().nbParts(stripBegin, stripBegin+nbPixels.first);
    }
    if (totalNbPart!=0u) {
        const uint32 totalPixels = nbPixels.first*nbPixels.second;
        HVector<thrust::pair<uint32,uint32>> dOffsets;
        mSph.reserveDevice(totalPixels, totalNbPart);
        for (uint32 cpt=0; cpt<nbPixels.second; cpt++) {
            const uint32 stripBegin = firstPixel+cpt*mWidth;
            dOffsets.push_back(mSph.hostToDevice(stripBegin, stripBegin+nbPixels.first));
        }
        DPtr<uint32> dWorldKeys{mSph.deviceWorlds().worldKey.get()};
        transform(dWorldKeys, dWorldKeys+mSph.allocatedParticles(), dWorldKeys,
            [=] DEVICE (uint32 value) {
                uint32 idx = value%mWidth;
                uint32 idy = value/mWidth;
                return (idy-startPixel.second)*nbPixels.first+(idx-startPixel.first);
            });
        mSph.setWorldBegin(0u);
        for (uint32 i=0; i<iterations; i++) {
            mSph.update(dt);
        }
        for (uint32 cpt=0; cpt<nbPixels.second; cpt++) {
            const uint32 stripBegin = firstPixel+cpt*mWidth;
            mSph.deviceToHost(stripBegin, stripBegin+nbPixels.first, dOffsets[cpt]);
        }
        mSph.cleanDevice();
    }
}
//*****************************************************************************
void Layer::convoluate(uint32 rows, uint32 cols, const thrust::pair<uint32,uint32>& nbSamples,
    bool debug)
{
    const uint32 nbPixels = rows*cols;
    mHostPixels.intensities.resize(nbPixels);

    const Vec2f ratio{mWidth/(float)cols, mHeight/(float)rows};
    //Pixels array, may be be aware of the case  when the output image is too large
    DUniqueArrayPtr<float> densities{cudaAllocator<float>(nbPixels)};
    const int32 initDensities = debug ? 1 : 0; 
    cudaFill(densities.get(), initDensities, nbPixels);
    auto& worlds = mSph.hostWorlds();
    uint32 startParticle = 0u;
    Vec2f pixelSize = mHostPixels.upperLimits[0]-mHostPixels.lowerLimits[0];
    pixelSize.x() *= ratio.x();
    pixelSize.y() *= ratio.y();
    DUniqueArrayPtr<uint32> dCellsStop{cudaAllocator<uint32>(nbPixels)};
    do {
        const uint32 nbParticles = allocableParticles(startParticle);
        DUniqueArrayPtr<Vec2f> dPos{cudaAllocator<Vec2f>(nbParticles)};
        cudaHostToDevice(worlds.particles.pos.data()+startParticle, dPos.get(), nbParticles);
        DUniqueArrayPtr<float> dRadius{cudaAllocator<float>(nbParticles)};
        cudaHostToDevice(worlds.particles.radius.data()+startParticle, dRadius.get(), nbParticles);
        //We compute particles locations in the output image
        computeGrid(dCellsStop, dPos, dRadius, {cols,rows}, pixelSize, nbParticles);
        DensityKernel{densities.get(), dPos.get(), dRadius.get(), dCellsStop.get(),
            nbSamples, mSph.params().maxRadius(), cols, rows, ratio, {0u,0u}, debug}();

        startParticle += nbParticles;
    } while(startParticle<worlds.particles.pos.size());

    if (!debug) {
        DPtr<float> thDensities{densities.get()};
        //We divide the result by the number of samples to compute the mean
        thrust::transform(thDensities, thDensities+nbPixels, thDensities,
            [=] DEVICE (float val) {return min(1., max(0., val/(nbSamples.first*nbSamples.second)));});
    }
    cudaDeviceToHost(densities.get(), mHostPixels.intensities.data(), nbPixels);
}
//*****************************************************************************
void Layer::convoluate(const thrust::pair<uint32,uint32>& startPixel, const thrust::pair<uint32,uint32>& nbPixels,
    uint32 rows, uint32 cols, const thrust::pair<uint32,uint32>& nbSamples, bool debug)
{
    const Vec2f ratio{nbPixels.first/(float)cols, nbPixels.second/(float)rows};
    const uint32 nbOutputPixels = rows*cols;
    const uint32 firstPixel = startPixel.second*mWidth+startPixel.first;
    mHostPixels.intensities.resize(nbOutputPixels);
    DUniqueArrayPtr<float> densities{cudaAllocator<float>(nbOutputPixels)};
    const int32 initDensities = debug ? 1 : 0; 
    cudaFill(densities.get(), initDensities, nbOutputPixels);
    Vec2f pixelSize = mHostPixels.upperLimits[0]-mHostPixels.lowerLimits[0];
    pixelSize.x() *= ratio.x();
    pixelSize.y() *= ratio.y();
    DUniqueArrayPtr<uint32> dCellsStop{cudaAllocator<uint32>(nbOutputPixels)};
    HVector<uint32> nbParts;
    //count number of particles
    uint32 totalNbPart = 0u;
    for (uint32 stripBegin = firstPixel; stripBegin<(startPixel.second+nbPixels.second)*mWidth; stripBegin+=mWidth) {
        nbParts.push_back(mSph.hostWorlds().nbParts(stripBegin, stripBegin+nbPixels.first));
        totalNbPart += nbParts.back();
    }
    if (totalNbPart!=0u) {
        //transfert data to GPU
        DUniqueArrayPtr<Vec2f> dPos{cudaAllocator<Vec2f>(totalNbPart)};
        DUniqueArrayPtr<float> dRadius{cudaAllocator<float>(totalNbPart)};
        uint32 deviceOffset = 0u;
        std::cerr << "cpt : " << nbPixels.second << std::endl;
        for (uint32 cpt=0; cpt<nbPixels.second; cpt++) {
            const uint32 stripBegin = firstPixel+cpt*mWidth;
            const uint32 firstPart = mSph.hostWorlds().partBegin(stripBegin);

            cudaHostToDevice(mSph.hostWorlds().particles.pos.data()+firstPart, dPos.get()+deviceOffset, nbParts[cpt]);
            cudaHostToDevice(mSph.hostWorlds().particles.radius.data()+firstPart, dRadius.get()+deviceOffset, nbParts[cpt]);
            deviceOffset += nbParts[cpt];
        }
        computeGrid(dCellsStop, dPos, dRadius, {cols,rows}, pixelSize, totalNbPart, startPixel);
        DensityKernel{densities.get(), dPos.get(), dRadius.get(), dCellsStop.get(),
            nbSamples, mSph.params().maxRadius(), cols, rows, ratio, startPixel, debug}();

        if (!debug) {
            DPtr<float> thDensities{densities.get()};
            //We divide the result by the number of samples to compute the mean
            thrust::transform(thDensities, thDensities+nbOutputPixels, thDensities,
                [=] DEVICE (float val) {return min(1., max(0., val/(nbSamples.first*nbSamples.second)));});
        }
        cudaDeviceToHost(densities.get(), mHostPixels.intensities.data(), nbOutputPixels);
    } else {
        thrust::fill(mHostPixels.intensities.begin(), mHostPixels.intensities.end(), 0.f);
    }
}
//*****************************************************************************
const SPH& Layer::SPH() const
{
    return mSph;
}
//*****************************************************************************
sph::SPH& Layer::SPH()
{
    return mSph;
}
//*****************************************************************************
void Layer::calculateGradientIntensity()
{
    cv::Mat input{(int32)mHeight, (int32)mWidth, CV_32F, 0.f};
    for(int32 i = 0; i < input.rows; i++) {
        float* row = input.ptr<float>(i);
        for(int32 j = 0; j < input.cols; j++) {
            row[j] = mHostPixels.intensities[i*mWidth+j];
        }
    }
    cv::GaussianBlur(input, input, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT);
    
    cv::Mat grad_x, grad_y;
    double scale = 4.;
	int32 delta = 0;
    int32 ddepth = CV_32F; // use 32 bits float to avoid overflow
    /// Gradient X
    cv::Sobel(input, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    /// Gradient Y
    cv::Sobel(input, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);

    for(int32 i=0;i<grad_x.rows;i++) {
        for(int32 j=0;j<grad_x.cols;j++) {
            Vec2d grad{grad_x.at<float>(i,j), -grad_y.at<float>(i,j)};
            mHostPixels.gradIntensities[i*mWidth+j][0] = grad.x();
            mHostPixels.gradIntensities[i*mWidth+j][1] = grad.y();
            mSph.setForce(i*mWidth+j, grad);
        }
    }
}
//*****************************************************************************
uint32 Layer::allocableParticles(uint32 startParticle)
{
    //particles positions and radius
    constexpr uint32 particleSize = sizeof(Vec2f)+sizeof(float);
    //We allocated all pixel intensities and the computed last particle index for each pixel
    const uint32 intensSize = mHostPixels.intensities.size()*sizeof(float);
    const uint32 cellsStopSize = mHostPixels.intensities.size()*sizeof(uint32);
    //We need grain radius for the kernel, image width and height,
    //samples number and ratio between input and output size
    constexpr uint32 paramsSize = sizeof(thrust::pair<uint32,uint32>)+sizeof(mSph.params().maxRadius())
        +2*sizeof(uint32)+sizeof(Vec2f);
    const uint32 availableSize = mProps.totalGlobalMem;
    
    const uint32 numerator = availableSize-paramsSize-intensSize-cellsStopSize;
    const uint32 denominator = particleSize;
    const uint32 maxParticle = numerator/denominator ;

    return min(maxParticle, (uint32)mSph.hostWorlds().particles.pos.size()-startParticle);
}
//*****************************************************************************
void Layer::computeGrid(DUniqueArrayPtr<uint32>& dCellsStop, DUniqueArrayPtr<Vec2f>& dPos,
    DUniqueArrayPtr<float>& dRadius, const Vec2<uint32>& dim, const Vec2f& pixelSize,
    uint32 partNumber, thrust::pair<uint32,uint32> startPixel)
{
    uint32 nbOutputPixels = dim.x()*dim.y();
    DUniqueArrayPtr<uint32> dOffset{cudaAllocator<uint32>(partNumber)};
    DUniqueArrayPtr<uint32> dPartByCell{cudaAllocator<uint32>(nbOutputPixels)};
    DUniqueArrayPtr<uint32> dIndexes{cudaAllocator<uint32>(partNumber)};
    GridKernel{dPos.get(), dIndexes.get(), dOffset.get(), dPartByCell.get(),
        partNumber, dim, pixelSize, startPixel}();

    //compute the last index in particle list for each grid cell
    DPtr<uint32> partByCell{dPartByCell.get()};
    DPtr<uint32> cellsStop{dCellsStop.get()};
    thrust::inclusive_scan(partByCell, partByCell+nbOutputPixels, cellsStop);

    //compute the new particles positions
    OffsetFunctor functor{dCellsStop.get()};
    DPtr<uint32> index{dIndexes.get()};
    DPtr<uint32> offset{dOffset.get()};
    thrust::transform(index, index+partNumber, offset, offset, functor);

    //sort particles
    DPtr<Vec2f> pos{dPos.get()};
    DUniqueArrayPtr<Vec2f> posResult{cudaAllocator<Vec2f>(partNumber)};
    DPtr<Vec2f> tPosResult{posResult.get()};
    thrust::scatter(pos, pos+partNumber, offset, tPosResult);
    dPos = std::move(posResult);

    DPtr<float> radius{dRadius.get()};
    DUniqueArrayPtr<float> radiusResult{cudaAllocator<float>(partNumber)};
    DPtr<float> tRadiusResult{radiusResult.get()};
    thrust::scatter(radius, radius+partNumber, offset, tRadiusResult);
    dRadius = std::move(radiusResult);
}
//*****************************************************************************
void Layer::copyInitOnHost(ConstPtr<float> dIntensities, ConstPtr<Vec2f> dCenters,
    ConstPtr<Vec2f> dLowers, ConstPtr<Vec2f> dUppers, ConstPtr<Vec2f> dPos, 
    ConstPtr<float> dMass, ConstPtr<float> dRadius, ConstPtr<uint32> dNbParts,
    ConstPtr<uint32> dWorldKeys, float h, uint32 nbPixels, uint32 nbParts)
{
    //copy results on host
    mHostPixels.intensities.resize(nbPixels);
    mHostPixels.gradIntensities.resize(nbPixels);
    mHostPixels.centers.resize(nbPixels);
    mHostPixels.lowerLimits.resize(nbPixels);
    mHostPixels.upperLimits.resize(nbPixels);
    auto& worlds = mSph.hostWorlds();
    cudaDeviceToHost(dIntensities, mHostPixels.intensities.data(), nbPixels);
    cudaDeviceToHost(dCenters, mHostPixels.centers.data(), nbPixels);
    cudaDeviceToHost(dLowers, mHostPixels.lowerLimits.data(), nbPixels);
    cudaDeviceToHost(dUppers, mHostPixels.upperLimits.data(), nbPixels);
    worlds.lowerLimits = mHostPixels.lowerLimits;
    worlds.upperLimits = mHostPixels.upperLimits;
    Vec2f gridDim{mTx/h, mTy/h};
    mSph.setGridDim({(uint32)ceil(gridDim.x()), (uint32)ceil(gridDim.y())});
    worlds.externalForces.resize(nbPixels);
    worlds.particles.reserve(nbParts);
    cudaDeviceToHost(dPos, worlds.particles.pos.data(), nbParts);
    thrust::fill(worlds.particles.vel.begin(), worlds.particles.vel.end(), Vec2f{0,0});
    cudaDeviceToHost(dMass, worlds.particles.mass.data(), nbParts);
    cudaDeviceToHost(dRadius, worlds.particles.radius.data(), nbParts);

    worlds.particlesByWorld.resize(nbPixels);
    cudaDeviceToHost(dNbParts, worlds.particlesByWorld.data(), nbPixels);
    worlds.worldKey.resize(nbParts);
    cudaDeviceToHost(dWorldKeys, worlds.worldKey.data(), nbParts);
}
}
