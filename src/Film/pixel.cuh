#pragma once
#include "utils/vec.cuh"
#include "utils/memory.cuh"

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace film {
    template <template<typename, typename...> typename Array, typename... Args>
    struct Pixels {
        virtual void reset() = 0;
        Array<float> intensities;
        Array<utils::Vec2f> gradIntensities;
        Array<utils::Vec2f> centers;
        Array<utils::Vec2f> lowerLimits;
        Array<utils::Vec2f> upperLimits;
    };

    struct HostPtrPixels : public Pixels<utils::HUniqueArrayPtr>
    {
        inline void reset() override;
    };
    
    struct DevicePtrPixels : public Pixels<utils::DUniqueArrayPtr>
    {
        inline void reset() override;
    };
    
    struct HostVectorPixels : public Pixels<utils::HVector>
    {
        inline void reset() override;
        inline void append(const HostVectorPixels& other);
    };

    struct Pixel {
        float intensity;
        utils::Vec2f gradItensity;
        utils::Vec2f center;
        utils::Vec2f lowerLimit;
        utils::Vec2f upperLimit;
    };
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace film {
//*****************************************************************************
void HostPtrPixels::reset()
{
    intensities.reset();
    gradIntensities.reset();
    centers.reset();
    lowerLimits.reset();
    upperLimits.reset();
}
//*****************************************************************************
void DevicePtrPixels::reset()
{
    intensities.reset();
    gradIntensities.reset();
    centers.reset();
    lowerLimits.reset();
    upperLimits.reset();
}
//*****************************************************************************
void HostVectorPixels::reset()
{
    intensities.clear();
    intensities.shrink_to_fit();
    gradIntensities.clear();
    gradIntensities.shrink_to_fit();
    centers.clear();
    centers.shrink_to_fit();
    lowerLimits.clear();
    lowerLimits.shrink_to_fit();
    upperLimits.clear();
    upperLimits.shrink_to_fit();
}
//*****************************************************************************
void HostVectorPixels::append(const HostVectorPixels& other)
{
    intensities.insert(intensities.end(), other.intensities.begin(), other.intensities.end());
    gradIntensities.insert(gradIntensities.end(), other.gradIntensities.begin(), other.gradIntensities.end());
    centers.insert(centers.end(), other.centers.begin(), other.centers.end());
    lowerLimits.insert(lowerLimits.end(), other.lowerLimits.begin(), other.lowerLimits.end());
    upperLimits.insert(upperLimits.end(), other.upperLimits.begin(), other.upperLimits.end());
}
}
