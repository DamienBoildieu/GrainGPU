#include "kernels.cuh"
#include "utils/filters.cuh"
#include "utils/normalDistribution.cuh"
#include "SPH/sphParameters.cuh"

using namespace utils;

namespace film {
//*****************************************************************************
enum class Color {
    red = 0u,
    green = 1u,
    blue = 2u
};
//*******************************************************************************
DEVICE
uint32 start(ConstPtr<uint32> cellsStop, int32 cell)
{
    return cell ? cellsStop[cell-1] : 0;
}
//*******************************************************************************
DEVICE
uint32 stop(ConstPtr<uint32> cellsStop, int32 cell)
{
    return cellsStop[cell];
}
//******************************************************************************
DEVICE
Vec2f getPos(uint32 pixel, uint32 width, Vec2f& ratio, const Vec2<uint32>& ite,
             const thrust::pair<uint32, uint32>& nbSamples)
{
    //pixel limits
    const Vec2f min{(pixel%width)*ratio.x(), (pixel/width)*ratio.y()};
    const Vec2f max = min + ratio;
    //linear interpolation, exclude pixel limits
    return {(max.x()-min.x())/(nbSamples.first+1.f)*(ite.x()+1.f)+min.x(),
            (max.y()-min.y())/(nbSamples.second+1.f)*(ite.y()+1.f)+min.y()};
}
//*****************************************************************************
DEVICE
double atomicMul(Ptr<double> address, double val)
{
    Ptr<unsigned long long int> address_as_ull =
       (Ptr<unsigned long long int>)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val *
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
//*****************************************************************************
DEVICE
float atomicMul(Ptr<float> address, float val)
{
    Ptr<unsigned int> address_as_ui =
        (Ptr<unsigned int>)address;
    unsigned int old = *address_as_ui, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ui, assumed,
                        __float_as_uint(val *
                               __float_as_uint(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __uint_as_float(old);
}
//*****************************************************************************
DEVICE
bool intersect(const Vec2f& grainCenter, float grainRadius, const Vec2f& target)
{
    Vec2f centered = target-grainCenter;
    //the location is covered if the square norm is lower or equal than the square radius
    //disk equation : (x-x0)^2+(y-y0)^2 <= R^2
    return centered.norm2() <= grainRadius*grainRadius;
}
//*******************************************************************************
GLOBAL
void initSPH(Ptr<float> intensities, Ptr<Vec2f> centers, Ptr<Vec2f> lowerLimits,
    Ptr<Vec2f> upperLimits, Ptr<Vec2f> pos, Ptr<float> mass, Ptr<float> radius,
    Ptr<uint32> worldKey, ConstPtr<Vec2f> possiblePos, ConstPtr<uint32> partStop,
    Ptr<curandState> states, Ptr<bool> mark, float intensityCoeff, float avgMass,
    float stdevMass, float rho0, Vec2<uint32> dim, uint32 maxPart)
{
    const uint32 startId = blockIdx.x*blockDim.x+threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=startId; tid<dim.x()*dim.y(); tid+=gridSize) {
        const uint32 partBegin = start(partStop, tid);
        curandState localState = states[tid];
        const float intensity = intensities[tid]*intensityCoeff;
        intensities[tid] = intensity;
        const Vec2f id{tid%dim.x(), tid/dim.x()};
        centers[tid] = id +.5f;
        lowerLimits[tid] = id;
        upperLimits[tid] = id+1.f;
        const uint32 nbParticles = intensity*maxPart;
        const uint32 markBegin = maxPart*tid;
        for (uint32 i=0; i<nbParticles; i++) {
            uint32 index;
            do {
                index = uint32(floor(curand_uniform(&localState)*(maxPart-i)));
            } while(mark[markBegin+index]);
            mark[markBegin+index] = true;
            const Vec2f posPart = id+possiblePos[index];
            pos[partBegin+i] = posPart;
            const float prob = curand_uniform(&localState);
            const float massPart = normalTruncMinCdfInv(prob, avgMass, stdevMass, 0.01);
            mass[partBegin+i] = massPart;
            radius[partBegin+i] = sph::radius(massPart, rho0);
            worldKey[partBegin+i] = tid;
        }
        states[tid] = localState;
    }
}
//******************************************************************************
GLOBAL
void initRandomPos(Ptr<float> intensities, Ptr<Vec2f> centers, Ptr<Vec2f> lowerLimits,
    Ptr<Vec2f> upperLimits, Ptr<curandState> states, Ptr<bool> mark,
    Ptr<uint32> nbPart, float intensityCoeff, Vec2<uint32> dim, uint32 maxPart)
{
    const uint32 startId = blockIdx.x*blockDim.x+threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=startId; tid<dim.x()*dim.y(); tid+=gridSize) {
        curandState localState = states[tid];
        const float intensity = intensities[tid]*intensityCoeff;
        intensities[tid] = intensity;
        const Vec2f id{tid%dim.x(), tid/dim.x()};
        centers[tid] = id +.5f;
        lowerLimits[tid] = id;
        upperLimits[tid] = id+1.f;
        const uint32 markBegin = maxPart*tid;
        uint32 nbParts = 0u;
        for (uint32 i=0; i<maxPart; i++){
            if (curand_uniform(&localState) <= intensity) {
                mark[markBegin+i] = true;
                nbParts++;
            }
        }
        nbPart[tid] = nbParts;
        states[tid] = localState;
    }
}
//******************************************************************************
GLOBAL
void initRandomSPH(Ptr<Vec2f> pos, utils::Ptr<float> mass, Ptr<float> radius,
    Ptr<uint32> worldKey, ConstPtr<Vec2f> possiblePos, ConstPtr<uint32> partStop,
    Ptr<curandState> states, ConstPtr<bool> mark, float avgMass,
    float stdevMass, float rho0, Vec2<uint32> dim, uint32 maxPart)
{
    const uint32 startId = blockIdx.x*blockDim.x+threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=startId; tid<dim.x()*dim.y(); tid+=gridSize) {
        curandState localState = states[tid];
        const Vec2f id{tid%dim.x(), tid/dim.x()};
        const uint32 markBegin = tid*maxPart;
        const uint32 partBegin = start(partStop, tid);
        uint32 nbPart = 0;
        for (uint32 i=0; i<maxPart; i++) {
            if (mark[markBegin+i]) {
                pos[partBegin+nbPart] = id+possiblePos[i];
                const float prob = curand_uniform(&localState);
                const float massPart = normalTruncMinCdfInv(prob, avgMass, stdevMass, 0.01);
                mass[partBegin+nbPart] = massPart;
                radius[partBegin+nbPart] = sph::radius(massPart, rho0);
                worldKey[partBegin+nbPart] = tid;
                nbPart++;
            }
        }
        states[tid] = localState;
    }
}
//*****************************************************************************
GLOBAL
void createGrid(ConstPtr<Vec2f> pos, Ptr<uint32> indexes, Ptr<uint32> offset,
    Ptr<uint32> partByCell, uint32 partNumber, Vec2<uint32> dim,
    Vec2f pixelSize, thrust::pair<uint32,uint32> firstPixel)
{
    const uint32 firstId = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=firstId; tid<partNumber; tid+=gridSize) {
        //particle position in final image
        const int32 idx = min(dim.x()-1,
            max(0,(int32) floor((pos[tid].x()-firstPixel.first)/pixelSize.x())));
        const int32 idy = min(dim.y()-1,
            max(0,(int32) floor((pos[tid].y()-firstPixel.second)/pixelSize.y())));
        const uint32 index = idx+idy*dim.x();
        indexes[tid] = index;
        offset[tid] = atomicAdd(&partByCell[index], 1u);
    }
}
//*****************************************************************************
GLOBAL
void computeDensity(Ptr<float> densities, ConstPtr<Vec2f> pos, ConstPtr<float> grainRadius,
    ConstPtr<uint32> cellsStop, thrust::pair<uint32, uint32> nbSamples, float maxGrainRadius,
    uint32 width, uint32 height, Vec2f ratio, thrust::pair<uint32, uint32> firstPixel, bool debug)
{
    const int32 nbcx = int32(ceilf(maxGrainRadius/ratio.x()));
    const int32 nbcy = int32(ceilf(maxGrainRadius/ratio.y()));
    const uint32 totalSamples = nbSamples.first*nbSamples.second;
    for (uint32 y=blockIdx.z*blockDim.z+threadIdx.z; y<height; y+=blockDim.z*gridDim.z) {
        for (uint32 x=blockIdx.y*blockDim.y+threadIdx.y; x<width; x+=blockDim.y*gridDim.y) {
            for (uint32 sample=blockIdx.x*blockDim.x+threadIdx.x; sample<totalSamples; sample+=blockDim.x*gridDim.x) {
                //compute the pixel index, we have width*height*nbIte elements
                //all samples of the same pixel are following each other
                uint32 pixel = y*width+x;
                //used to interpolate position, the interpolation x index and y index
                Vec2<uint32> posPixelIdx{sample%nbSamples.first, sample/nbSamples.first};
                Vec2f samplePos = getPos(pixel, width, ratio, posPixelIdx, nbSamples);
                samplePos.x() += firstPixel.first;
                samplePos.y() += firstPixel.second;
                //the pixel index
                uint32 idx = pixel%width;
                uint32 idy = pixel/width;

                bool intersected = false;
                int32 i = idx-nbcx;
                Color color;
                //iterate over neighbors pixels
                while (i<=(int32)idx+nbcx && !intersected){
                    int32 j = idy-nbcy;
                    while (j<=(int32)idy+nbcy && !intersected) {
                        if(i>=0 && j>=0 && i<width && j<height){
                            const int32 cell = i + j*width;
                            //iterate over particles
                            for (uint32 grainId=start(cellsStop, cell); grainId<stop(cellsStop, cell) && !intersected; grainId++) {
                                const Vec2f grainPos = pos[grainId];
                                const float radius = grainRadius[grainId];
                                //compute intersection between the grain and the sample
                                intersected = intersect(grainPos, radius, samplePos);
                                if (debug && intersected)
                                    color = static_cast<Color>(grainId%3);
                            }
                        }
                        j++;
                    }
                    i++;
                }
                //if a grain is located at the position, add its contribution
                if (intersected) {
                    if (debug) {
                        switch (color) {
                            case Color::red:
                                atomicMul(&densities[pixel], 2.f);
                                break;
                            case Color::green:
                                atomicMul(&densities[pixel], 3.f);
                                break;
                            case Color::blue:
                                atomicMul(&densities[pixel], 5.f);
                                break;
                        }
                    } else
                        atomicAdd(&densities[pixel], 1.f);
                }
            }
        }
    }
}
//*****************************************************************************
GLOBAL
void initRandomState(Ptr<curandState> states, uint64 seed, uint32 nbStates)
{
    const uint32 startId = blockIdx.x*blockDim.x+threadIdx.x;
    const uint32 gridSize = blockDim.x * gridDim.x;
    for (uint32 tid=startId; tid<nbStates; tid+=gridSize)
        curand_init(seed, tid, 0, &states[tid] );
}
}
