#pragma once
#include "utils/vec.cuh"
#include "utils/memory.cuh"

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace sph {
struct Particle {
    utils::Vec2f pos;
    utils::Vec2f vel;
    utils::Vec2d force;
    float rho;
    float p;
    float mass;
    float radius;
};

template <template<typename, typename...> typename Array, typename... Args>
struct Particles {
    virtual void reset() = 0;
    virtual void reserve(uint32 nbElems) = 0;
    
    Array<utils::Vec2f> pos;
    Array<utils::Vec2f> vel;
    Array<utils::Vec2d> force;
    Array<float> rho;
    Array<float> p;
    Array<float> mass;
    Array<float> radius;
};

struct DevicePtrParticles;

struct HostPtrParticles : public Particles<utils::HUniqueArrayPtr>
{
    inline void reset() override;
    inline void reserve(uint32 nbElems) override;
};

struct HostVectorParticles : public Particles<utils::HVector>
{
    inline void reset() override;
    inline void reserve(uint32 nbElems) override;
    inline void copyFromDevice(const DevicePtrParticles& device, uint32 hOffset,
        uint32 nbElems, uint32 dOffset=0u);
    inline void append(const HostVectorParticles& other);
    inline void append(const Particle& particle);
};

struct DevicePtrParticles : public Particles<utils::DUniqueArrayPtr>
{
    inline void reset() override;
    inline void reserve(uint32 nbElems) override;
    inline void copyFromHost(const HostVectorParticles& host, uint32 hOffset,
        uint32 nbElems, uint32 dOffset=0u);
};
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace sph {
//*****************************************************************************
void HostPtrParticles::reset()
{
    pos.reset();
    vel.reset();
    force.reset();
    rho.reset();
    p.reset();
    mass.reset();
    radius.reset();
}
//*****************************************************************************
void HostPtrParticles::reserve(uint32 nbElems)
{
    pos.reset(new utils::Vec2f[nbElems]);
    vel.reset(new utils::Vec2f[nbElems]);
    force.reset(new utils::Vec2d[nbElems]);
    rho.reset(new float[nbElems]);
    p.reset(new float[nbElems]);
    mass.reset(new float[nbElems]);
    radius.reset(new float[nbElems]);
}
//*****************************************************************************
void HostVectorParticles::reset()
{
    pos.clear();
    pos.shrink_to_fit();
    vel.clear();
    vel.shrink_to_fit();
    force.clear();
    force.shrink_to_fit();
    rho.clear();
    rho.shrink_to_fit();
    p.clear();
    p.shrink_to_fit();
    mass.clear();
    mass.shrink_to_fit();
    radius.clear();
    radius.shrink_to_fit();
}
//*****************************************************************************
void HostVectorParticles::reserve(uint32 nbElems)
{
    pos.resize(nbElems);
    vel.resize(nbElems);
    force.resize(nbElems);
    rho.resize(nbElems);
    p.resize(nbElems);
    mass.resize(nbElems);
    radius.resize(nbElems);
}
//*****************************************************************************
void HostVectorParticles::copyFromDevice(const DevicePtrParticles& device, uint32 hOffset,
    uint32 nbElems, uint32 dOffset)
{
    utils::cudaDeviceToHost(device.pos.get()+dOffset, pos.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.vel.get()+dOffset, vel.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.force.get()+dOffset, force.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.rho.get()+dOffset, rho.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.p.get()+dOffset, p.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.mass.get()+dOffset, mass.data()+hOffset, nbElems);
    utils::cudaDeviceToHost(device.radius.get()+dOffset, radius.data()+hOffset, nbElems);
}
//*****************************************************************************
void HostVectorParticles::append(const HostVectorParticles& other)
{
    pos.insert(pos.end(), other.pos.begin(), other.pos.end());
    vel.insert(vel.end(), other.vel.begin(), other.vel.end());
    force.insert(force.end(), other.force.begin(), other.force.end());
    rho.insert(rho.end(), other.rho.begin(), other.rho.end());
    p.insert(p.end(), other.p.begin(), other.p.end());
     mass.insert(mass.end(), other.mass.begin(), other.mass.end());
    radius.insert(radius.end(), other.radius.begin(), other.radius.end());
}
//*****************************************************************************
void HostVectorParticles::append(const Particle& particle)
{
    pos.push_back(particle.pos);
    vel.push_back(particle.vel);
    force.push_back(particle.force);
    rho.push_back(particle.rho);
    p.push_back(particle.p);
    mass.push_back(particle.mass);
    radius.push_back(particle.radius);
}
//*****************************************************************************
void DevicePtrParticles::reset()
{
    pos.reset();
    vel.reset();
    force.reset();
    rho.reset();
    p.reset();
    mass.reset();
    radius.reset();
}
//*****************************************************************************
void DevicePtrParticles::reserve(uint32 nbElems)
{
    pos.reset(utils::cudaAllocator<utils::Vec2f>(nbElems));
    vel.reset(utils::cudaAllocator<utils::Vec2f>(nbElems));
    force.reset(utils::cudaAllocator<utils::Vec2d>(nbElems));
    rho.reset(utils::cudaAllocator<float>(nbElems));
    p.reset(utils::cudaAllocator<float>(nbElems));
    mass.reset(utils::cudaAllocator<float>(nbElems));
    radius.reset(utils::cudaAllocator<float>(nbElems));
}
//*****************************************************************************
void DevicePtrParticles::copyFromHost(const HostVectorParticles& host, uint32 hOffset,
    uint32 nbElems, uint32 dOffset)
{
    utils::cudaHostToDevice(host.pos.data()+hOffset, pos.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.vel.data()+hOffset, vel.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.force.data()+hOffset, force.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.rho.data()+hOffset, rho.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.p.data()+hOffset, p.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.mass.data()+hOffset, mass.get()+dOffset, nbElems);
    utils::cudaHostToDevice(host.radius.data()+hOffset, radius.get()+dOffset, nbElems);
}
}
