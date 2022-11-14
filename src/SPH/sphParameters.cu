#include "sphParameters.cuh"
#include <cmath>
#include <iostream>

namespace sph {
//*****************************************************************************
HOST DEVICE
float radius(float mass, float rho0)
{
    return cbrt(60.f*mass/(4.f*M_PI*rho0));
}
//*****************************************************************************
SPHParameters::SPHParameters(float avgMass, float massStdev, float minMass, float maxMass,
    float rho0, float mu, float elast, float fric)
: mAvgMass(avgMass), mMassStdev(massStdev), mMinMass(minMass), mMaxMass(maxMass),
  mRho0(rho0), mMu(mu), mElast(elast), mFric(fric)
{
    mAvgRadius = sph::radius(mAvgMass, mRho0);
    mMinRadius = sph::radius(mMinMass, mRho0);
    mMaxRadius = sph::radius(mMaxMass, mRho0);
    mTaitsB = mRho0 * 88.5f * 88.5f / (7.f * 1000.f);
}
//*****************************************************************************
float SPHParameters::avgMass() const
{ 
    return mAvgMass;//0.5f;
}
//*****************************************************************************
void SPHParameters::setAvgMass(float avgMass)
{
    mAvgMass = avgMass;
    mAvgRadius = sph::radius(mAvgMass, mRho0);
}
//*****************************************************************************
float SPHParameters::massStdev() const
{
    return mMassStdev;
}
//*****************************************************************************
void SPHParameters::setMassStdev(float massStdev)
{
    mMassStdev = massStdev;
}
//*****************************************************************************
float SPHParameters::minMass() const
{
    return mMinMass;
}
//*****************************************************************************
void SPHParameters::setMinMass(float minMass)
{
    mMinMass = minMass;
    mMinRadius = sph::radius(mMinMass, mRho0);
}
//*****************************************************************************
float SPHParameters::maxMass() const
{
    return mMaxMass;
}
//*****************************************************************************
void SPHParameters::setMaxMass(float maxMass)
{
    mMaxMass = maxMass;
    mMaxRadius = sph::radius(mMaxMass, mRho0);
}
//*****************************************************************************
float SPHParameters::rho0() const
{ 
    return mRho0;
}
//*****************************************************************************
void SPHParameters::setRho0(float rho0)
{
    mRho0 = rho0;
    mAvgRadius = sph::radius(mAvgMass, mRho0);
    mMinRadius = sph::radius(mMinMass, mRho0);
    mMaxRadius = sph::radius(mMaxMass, mRho0);
    mTaitsB = mRho0 * 88.5f * 88.5f / (7.f * 1000.f);
}
//*****************************************************************************
float SPHParameters::avgRadius() const
{
    return mAvgRadius;
}
//*****************************************************************************
float SPHParameters::minRadius() const
{
    return mMinRadius;
}
//*****************************************************************************
float SPHParameters::maxRadius() const
{
    return mMaxRadius;
}
//*****************************************************************************
float SPHParameters::mu() const
{ 
    return mMu;
}
//*****************************************************************************
void SPHParameters::setMu(float mu)
{
    mMu = mu;
}
//*****************************************************************************
float SPHParameters::taitsB() const
{ 
    return mTaitsB;
}
//*****************************************************************************
float SPHParameters::elast() const
{ 
    return mElast;
}
//*****************************************************************************
void SPHParameters::setElast(float elast)
{
    mElast = elast;
}
//*****************************************************************************
float SPHParameters::fric() const
{ 
    return mFric;
}
//*****************************************************************************
void SPHParameters::setFric(float fric)
{
    mFric = fric;
}
}
