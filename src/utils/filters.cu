#include "filters.cuh"

namespace utils {
//*****************************************************************************
HOST DEVICE BoxFilter::BoxFilter(const Vec2f& radius)
: Filter(radius)
{}
//*****************************************************************************
HOST DEVICE
float BoxFilter::filter(const Vec2f& point) const
{
    return boxFilter(point.x(), radius.x())*boxFilter(point.y(), radius.y());
}
//*****************************************************************************
HOST DEVICE
float BoxFilter::integrate(const Vec2f& max, const Vec2f& min) const
{
    return (boxPrimitive(max.x())-boxPrimitive(min.x()))*
            (boxPrimitive(max.y())-boxPrimitive(min.y()));
}
//*****************************************************************************
HOST DEVICE GaussianFilter::GaussianFilter(const Vec2f& radius)
: Filter(radius)
{}
//*****************************************************************************
HOST DEVICE
float GaussianFilter::filter(const Vec2f& point) const
{
    return gaussianFilter(point.x(), expRadius.x()) * gaussianFilter(point.y(), expRadius.y());
}
//*****************************************************************************
HOST DEVICE
float GaussianFilter::integrate(const Vec2f& max, const Vec2f& min) const
{
    float maxIntegralX = gaussianPrimitive(max.x(), expRadius.x());
    float maxIntegralY = gaussianPrimitive(max.y(), expRadius.y());
    float minIntegralX = gaussianPrimitive(min.x(), expRadius.x());
    float minIntegralY = gaussianPrimitive(min.y(), expRadius.y());

    float integral = (maxIntegralX-minIntegralX)*(maxIntegralY*minIntegralY);

    return integral;
}
//*****************************************************************************
HOST DEVICE MitchellFilter::MitchellFilter(const Vec2f& radius)
: Filter(radius)
{}
//*****************************************************************************
HOST DEVICE
float MitchellFilter::filter(const Vec2f& point) const
{
    return mitchellFilter(point.x(), radius.x())*mitchellFilter(point.y(), radius.y());
}
//*****************************************************************************
HOST DEVICE
float MitchellFilter::integrate(const Vec2f& max, const Vec2f& min) const
{
    float maxIntegralX = gaussianPrimitive(max.x(), radius.x());
    float maxIntegralY = gaussianPrimitive(max.y(), radius.y());
    float minIntegralX = gaussianPrimitive(min.x(), radius.x());
    float minIntegralY = gaussianPrimitive(min.y(), radius.y());

    float integral = (maxIntegralX-minIntegralX)*(maxIntegralY*minIntegralY);

    return integral;
}
//*****************************************************************************
HOST DEVICE
float boxFilter(float x, float radius)
{
    return abs(x)<=radius? 1. : 0.;
}
//*****************************************************************************
HOST DEVICE
float boxPrimitive(float x)
{
    return x;
}
//*****************************************************************************
HOST DEVICE
float gaussianFilter(float x, float expRadius)
{
    return max(0.,expf(-alpha*x*x)-expRadius);
}
//*****************************************************************************
/*
 * TODO Implement primitive
 */
HOST DEVICE
float gaussianPrimitive(float x, float expRadius)
{
    return expf(-alpha*x*x)-(expRadius*x);
}
//*****************************************************************************
HOST DEVICE
float mitchellFilter(float x, float radius)
{
    x = abs(2*(x/radius));
    if (x>1)
        return ((-B-6*C)*x*x*x + (6*B+30*C)*x*x + (-12*B-48*C) * x + (8*B+24*C)) * (1.f/6.f);
    else
        return ((12-9*B-6*C)*x*x*x + (-18+12*B+6*C)*x*x + (6-2*B)) * (1.f/6.f);
}
//*****************************************************************************
/*
 * Seems doesn't work
 */
HOST DEVICE
float mitchellPrimitive(float x, float radius)
{
    x = abs(2*(x/radius));
    if (x>1)
        return (((-B-6*C)*x*x*x*x)/4. + ((6*B+30*C)*x*x*x)/3. + ((-12*B-48*C)*x*x)/2. + (8*B+24*C)*x) * (1.f/6.f);
    else
        return (((12-9*B-6*C)*x*x*x*x)/4. + ((-18+12*B+6*C)*x*x*x)/3. + (6-2*B)*x) * (1.f/6.f);
}
}
