#pragma once
#include "utils/cuda.cuh"

namespace sph {
HOST DEVICE float radius(float mass, float rho0);
/**
 * @brief The SPHParameters struct
 * Define mass, rho0, radius, mus, taits, elasticiy and friction coefficients
 */
class SPHParameters {
public:
    SPHParameters() = default;
    SPHParameters(float avgMass, float massStdev, float minMass, float maxMass,
        float rho0, float mu, float elast, float fric);
    SPHParameters(const SPHParameters& other) = default;
    SPHParameters(SPHParameters&& other) = default;
    virtual ~SPHParameters() = default;
    SPHParameters& operator=(const SPHParameters& right) = default;
    SPHParameters& operator=(SPHParameters&& right) = default;

    float avgMass() const;
    void setAvgMass(float avgMass);
    float massStdev() const;
    void setMassStdev(float massStdev);
    float minMass() const;
    void setMinMass(float minMass);
    float maxMass() const;
    void setMaxMass(float maxMass);
    float rho0() const;
    void setRho0(float rho0);
    //theorical avg, min and max
    float avgRadius() const;
    float minRadius() const;
    float maxRadius() const;
    float mu() const;
    void setMu(float mu);
    float taitsB() const;
    float elast() const;
    void setElast(float elast);
    float fric() const;
    void setFric(float fric);
private:
    float mAvgMass;
    float mMassStdev;
    float mMinMass;
    float mMaxMass;
    float mAvgRadius;
    float mMinRadius;
    float mMaxRadius;
    float mRho0;
    float mMu;
    float mTaitsB;
    float mElast;
    float mFric;
};
}