#pragma once
#include "utils/vec.cuh"
#include "utils/memory.cuh"

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace sph {
template <template<typename, typename...> typename Array, typename... Args>
struct Grids {
    virtual void reset() = 0;
    constexpr uint32 particleDependSize() const;
    constexpr uint32 dimDependSize() const;
    constexpr uint32 constSize() const;

    Array<uint32> index;
    Array<uint32> offset;
    
    Array<uint32> partByCell;
    Array<uint32> cellsStop;
    
    utils::Vec2<uint32> dims;
};

struct HostPtrGrids : public Grids<utils::HUniqueArrayPtr>
{
    inline void reset() override;
};

struct HostVectorGrids;

struct DevicePtrGrids : public Grids<utils::DUniqueArrayPtr>
{
    inline void reset() override;
    inline void reserve(uint32 nbWorld, uint32 nbPart);
};

struct HostVectorGrids : public Grids<utils::HVector>
{
    inline void reset() override;
};
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace sph {
//*****************************************************************************
template <template<typename, typename...> typename Array, typename... Args>
constexpr uint32 Grids<Array, Args...>::particleDependSize() const
{
    //index and offset types
    return 2*sizeof(uint32);
}
//*****************************************************************************
template <template<typename, typename...> typename Array, typename... Args>
constexpr uint32 Grids<Array, Args...>::dimDependSize() const
{
    //partsByCell and cellsStop types
    return 2*sizeof(uint32);
}
//*****************************************************************************
template <template<typename, typename...> typename Array, typename... Args>
constexpr uint32 Grids<Array, Args...>::constSize() const
{
    //dim type
    return 2*sizeof(uint32);
}
//*****************************************************************************
void HostPtrGrids::reset()
{
    index.reset();
    offset.reset();
    partByCell.reset();
    cellsStop.reset();
}
//*****************************************************************************
void DevicePtrGrids::reset()
{
    index.reset();
    offset.reset();
    partByCell.reset();
    cellsStop.reset();
}
//*****************************************************************************
void DevicePtrGrids::reserve(uint32 nbWorld, uint32 nbPart)
{
    index.reset(utils::cudaAllocator<uint32>(nbPart));
    offset.reset(utils::cudaAllocator<uint32>(nbPart));
    std::cerr << "reserve " << nbWorld*dims.x()*dims.y() << std::endl;
    partByCell.reset(utils::cudaAllocator<uint32>(nbWorld*dims.x()*dims.y()));
    cellsStop.reset(utils::cudaAllocator<uint32>(nbWorld*dims.x()*dims.y()));
}
//*****************************************************************************
void HostVectorGrids::reset()
{
    index.clear();
    index.shrink_to_fit();
    offset.clear();
    offset.shrink_to_fit();
    partByCell.clear();
    partByCell.shrink_to_fit();
    cellsStop.clear();
    cellsStop.shrink_to_fit();
}
}
