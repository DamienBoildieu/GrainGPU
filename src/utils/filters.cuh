#pragma once
#include "vec.cuh"

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace utils {
constexpr float alpha = 1.;
constexpr float B = 1./3.;
constexpr float C = 1./3.;

template <typename Impl>
struct Filter {
    HOST DEVICE Filter(const Vec2f& radius);
    Filter(const Filter& other) = default;
    Filter(Filter&& other) = default;
    ~Filter() = default;
    Filter& operator=(const Filter& right) = default;
    Filter& operator=(Filter&& right) = default;

    HOST DEVICE float filter(const Vec2f& point) const;
    HOST DEVICE float integrate(const Vec2f& max, const Vec2f& min) const;

    Vec2f radius;
private:
    HOST DEVICE const Impl& impl() const;
    HOST DEVICE Impl& impl();
};

struct BoxFilter : public Filter<BoxFilter>
{
    HOST DEVICE BoxFilter(const Vec2f& radius);
    BoxFilter(const BoxFilter& other) = default;
    BoxFilter(BoxFilter&& other) = default;
    ~BoxFilter() = default;
    BoxFilter& operator=(const BoxFilter& right) = default;
    BoxFilter& operator=(BoxFilter&& right) = default;

    HOST DEVICE float filter(const Vec2f& point) const;
    HOST DEVICE float integrate(const Vec2f& max, const Vec2f& min) const;
};

struct GaussianFilter : public Filter<GaussianFilter>
{
    HOST DEVICE GaussianFilter(const Vec2f& radius);
    GaussianFilter(const GaussianFilter& other) = default;
    GaussianFilter(GaussianFilter&& other) = default;
    ~GaussianFilter() = default;
    GaussianFilter& operator=(const GaussianFilter& right) = default;
    GaussianFilter& operator=(GaussianFilter&& right) = default;

    HOST DEVICE float filter(const Vec2f& point) const;
    HOST DEVICE float integrate(const Vec2f& max, const Vec2f& min) const;

    Vec2f expRadius;
};

struct MitchellFilter : public Filter<MitchellFilter>
{
    HOST DEVICE MitchellFilter(const Vec2f& radius);
    MitchellFilter(const MitchellFilter& other) = default;
    MitchellFilter(MitchellFilter&& other) = default;
    ~MitchellFilter() = default;
    MitchellFilter& operator=(const MitchellFilter& right) = default;
    MitchellFilter& operator=(MitchellFilter&& right) = default;

    HOST DEVICE float filter(const Vec2f& point) const;
    HOST DEVICE float integrate(const Vec2f& max, const Vec2f& min) const;
};

HOST DEVICE float boxFilter(float x, float radius);
HOST DEVICE float boxPrimitive(float x);
HOST DEVICE float gaussianFilter(float x, float expRadius);
HOST DEVICE float gaussianPrimitive(float x, float expRadius);
HOST DEVICE float mitchellFilter(float x, float radius);
HOST DEVICE float mitchellPrimitive(float x, float radius);
}
//*****************************************************************************
//Definitions
namespace utils {
//*****************************************************************************
template <typename Impl>
HOST DEVICE Filter<Impl>::Filter(const Vec2f& radius)
: radius(radius)
{}
//*****************************************************************************
template <typename Impl>
HOST DEVICE float Filter<Impl>::filter(const Vec2f& point) const
{
    return impl().filter(point);
}
//*****************************************************************************
template <typename Impl>
HOST DEVICE float Filter<Impl>::integrate(const Vec2f& max, const Vec2f& min) const
{
    return impl().integrate(max, min);
}
//*****************************************************************************
template <typename Impl>
HOST DEVICE const Impl& Filter<Impl>::impl() const
{
    return static_cast<const Impl&>(*this);
}
//*****************************************************************************
template <typename Impl>
HOST DEVICE Impl& Filter<Impl>::impl()
{
    return static_cast<Impl&>(*this);
}
}
