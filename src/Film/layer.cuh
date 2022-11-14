#pragma once
#include <iostream>
#include <utility>
#include <random>
#include <vector>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/pair.h>
#include "pixel.cuh"
#include "SPH/sph.cuh"
#include "utils/cuda.cuh"
#include "utils/memory.cuh"
#include "utils/functors.cuh"
#include "kernelLauncher.cuh"
//*****************************************************************************
//Declaration
//*****************************************************************************
namespace film {
class Layer 
{
public:
    //Layer() = default;
    Layer(const Layer& other);
    template <typename Iterator>
    Layer(uint32 width, unsigned height, float intensityCoeff, Iterator intensity,
        const sph::SPHParameters& params, bool writeResults=false, bool writeForceStats=false);
    virtual ~Layer() = default;
    Layer& operator=(const Layer& right);

    const HostVectorPixels& hostPixels() const;
    uint32 width() const;
    uint32 height() const;

    void setAvgMass(float avgMass);
    void setMassStdev(float massStdev);
    void setRho0(float rho0);
    void setMu(float mu);
    void setElast(float elast);
    void setFric(float fric);
    void setSimuParams(const sph::SPHParameters& parameters);

    void setWriteResults(bool writeResults);
    void setWriteForceStats(bool forceStats);

    void update(float dt, uint32 iterations=1u);
    void update(const thrust::pair<uint32,uint32>& startPixel, const thrust::pair<uint32,uint32>& nbPixels,
        float dt, uint32 iterations);
    //Recontruct the input image with a convolution of grains at the desired resolution
    void convoluate(uint32 rows, uint32 cols, const thrust::pair<uint32,uint32>& nbSamples, bool debug);
    //Recontruct a part of the input image with a convolution of grains at the desired resolution
    void convoluate(const thrust::pair<uint32,uint32>& startPixel, const thrust::pair<uint32,uint32>& nbPixels,
                    uint32 rows, uint32 cols, const thrust::pair<uint32,uint32>& nbSamples, bool debug);

    const sph::SPH& SPH() const;
    sph::SPH& SPH();
    
private:
    cudaDeviceProp mProps;
    sph::SPH mSph;
    HostVectorPixels mHostPixels;
    uint32 mWidth;
    uint32 mHeight;
    float mIntensityCoeff;
    float mTx;
    float mTy;

    template <typename Iterator>
    void initLayerOnDevice(Iterator intensity);
    void calculateGradientIntensity();
    uint32 allocableParticles(uint32 startParticle);
    void computeGrid(utils::DUniqueArrayPtr<uint32>& dCellsStop, utils::DUniqueArrayPtr<utils::Vec2f>& dPos,
        utils::DUniqueArrayPtr<float>& dRadius, const utils::Vec2<uint32>& dim, const utils::Vec2f& pixelSize,
        uint32 partNumber, thrust::pair<uint32,uint32> startPixel={0u,0u});
    void copyInitOnHost(utils::ConstPtr<float> dIntensities, utils::ConstPtr<utils::Vec2f> dCenters,
        utils::ConstPtr<utils::Vec2f> dLowers, utils::ConstPtr<utils::Vec2f> dUppers,
        utils::ConstPtr<utils::Vec2f> dPos, utils::ConstPtr<float> dMass, utils::ConstPtr<float> dRadius,
        utils::ConstPtr<uint32> dNbParts, utils::ConstPtr<uint32> dWorldKeys, float h,
        uint32 nbPixels, uint32 nbParts);
};
}
//*****************************************************************************
//Definition
//*****************************************************************************
namespace film {
//*****************************************************************************
template <typename Iterator>
Layer::Layer(uint32 width, unsigned height, float intensityCoeff, Iterator intensity,
   const sph::SPHParameters& params,bool writeResults, bool writeForceStats)
: mProps(), mSph(params, writeResults, writeForceStats), mHostPixels(), mWidth(width),
  mHeight(height), mIntensityCoeff(intensityCoeff), mTx(1.f), mTy(1.f)
{
    HANDLE_ERROR(cudaGetDeviceProperties(&mProps, utils::chooseBestDevice()));
    initLayerOnDevice(intensity);
    calculateGradientIntensity();
}
//*****************************************************************************
template <typename Iterator>
void Layer::initLayerOnDevice(Iterator intensity)
{
    //prepare init
    const uint32 nbPixels = mWidth * mHeight;
    const float h = mSph.params().avgRadius();
    const uint32 nbIte = floor(1+(1-0.2*h*mTx)/(0.4*h*mTx))
            * floor((1+(1-0.2*h*mTy)/(0.4*h*mTy)));
    
    utils::DVector<float> dIntensities{intensity, intensity+nbPixels};
    utils::DUniqueArrayPtr<utils::Vec2f> dCenters{utils::cudaAllocator<utils::Vec2f>(nbPixels)};
    utils::DUniqueArrayPtr<utils::Vec2f> dLowers{utils::cudaAllocator<utils::Vec2f>(nbPixels)};
    utils::DUniqueArrayPtr<utils::Vec2f> dUppers{utils::cudaAllocator<utils::Vec2f>(nbPixels)};
    utils::DUniqueArrayPtr<uint32> dNbParts{utils::cudaAllocator<uint32>(nbPixels)};
    utils::DPtr<uint32> tNbParts{dNbParts.get()};

    utils::DUniqueArrayPtr<curandState> dStates{utils::cudaAllocator<curandState>(nbPixels)};
    uint32 seed = 0u;//std::chrono::system_clock::now().time_since_epoch().count();
    RandomKernel(dStates.get(), seed, nbPixels)();

    utils::DUniqueArrayPtr<bool> dMarks{utils::cudaAllocator<bool>(nbPixels*nbIte)};
    
    //RandomPosKernel{dIntensities.data().get(), dCenters.get(), dLowers.get(), dUppers.get(),
    //    dStates.get(), dMarks.get(), dNbParts.get(), mIntensityCoeff, {mWidth, mHeight},
    //    nbIte}();

    utils::cudaFill(dMarks.get(), 0, nbPixels*nbIte);
    thrust::transform(dIntensities.begin(), dIntensities.end(), tNbParts,
        utils::NbPartFunctor{nbIte, mIntensityCoeff});

    utils::DPtr<uint32> dPartsEnd{utils::cudaAllocator<uint32>(nbPixels)};
    utils::DPtr<uint32> tPartsEnd{dPartsEnd.get()};
    thrust::inclusive_scan(tNbParts, tNbParts+nbPixels, tPartsEnd);
    uint32 nbParts;
    utils::cudaDeviceToHost(dPartsEnd.get()+(nbPixels-1), &nbParts, 1);

    utils::HVector<utils::Vec2f> possiblePos;
    for(float x=0.2*h*mTx; x<=mTx; x+=0.4*h*mTx){
        for(float y=0.2*h*mTy; y<=mTy; y+=0.4*h*mTy){
            possiblePos.push_back({x,y});
        }
    }

    utils::DUniqueArrayPtr<utils::Vec2f> dPos{utils::cudaAllocator<utils::Vec2f>(nbParts)};
    utils::DUniqueArrayPtr<float> dMass{utils::cudaAllocator<float>(nbParts)};
    utils::DUniqueArrayPtr<float> dRadius{utils::cudaAllocator<float>(nbParts)};
    utils::DVector<utils::Vec2f> dPossiblesPos{possiblePos};
    utils::DUniqueArrayPtr<uint32> dWorldKeys{utils::cudaAllocator<uint32>(nbParts)};

    //InitRandomKernel{dPos.get(), dMass.get(), dRadius.get(), dWorldKeys.get(),
    //    dPossiblesPos.data().get(), dPartsEnd.get(), dStates.get(), dMarks.get(),
    //    mSph.params().avgMass(), mSph.params().massStdev(), mSph.params().rho0(),
    //    {mWidth, mHeight}, nbIte}();
    InitKernel(dIntensities.data().get(), dCenters.get(), dLowers.get(), dUppers.get(),
        dPos.get(), dMass.get(), dRadius.get(), dWorldKeys.get(), dPossiblesPos.data().get(),
        dPartsEnd.get(), dStates.get(), dMarks.get(), mIntensityCoeff, mSph.params().avgMass(),
        mSph.params().massStdev(), mSph.params().rho0(), {mWidth, mHeight}, nbIte)();

    //copy results on host
    copyInitOnHost(dIntensities.data().get(), dCenters.get(), dLowers.get(),
        dUppers.get(), dPos.get(), dMass.get(), dRadius.get(), dNbParts.get(),
        dWorldKeys.get(), h, nbPixels, nbParts);
}
}
