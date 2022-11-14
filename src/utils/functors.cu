#include "functors.cuh"

namespace utils {
//*****************************************************************************
OffsetFunctor::OffsetFunctor(uint32 const * const cellsStop)
: cellsStop(cellsStop)
{}
//*****************************************************************************
HOST DEVICE
uint32 OffsetFunctor::operator()(uint32 index, uint32 offset)
{
    return offset + (index==0 ? 0 : cellsStop[index-1]);
}
//*****************************************************************************
IntegrateFunctor::IntegrateFunctor(float dt)
: dt(dt)
{}
//*****************************************************************************
HOST DEVICE
Vec2f IntegrateFunctor::operator()(const Vec2f& left, const Vec2f& right)
{
    return left + right*dt;
}
//*****************************************************************************
NbPartFunctor::NbPartFunctor(uint32 nbIte, float intensityCoeff)
: nbIte(nbIte), intensityCoeff(intensityCoeff)
{}
//*****************************************************************************
HOST DEVICE
uint32 NbPartFunctor::operator()(float val)
{
    return val * intensityCoeff * nbIte;
}
//*****************************************************************************
HOST DEVICE
Vec2d MinFunctor::operator()(const Vec2d& left, const Vec2d& right)
{
    return left.norm2() < right.norm2() ? left : right;
}
//*****************************************************************************
HOST DEVICE
Vec2d MaxFunctor::operator()(const Vec2d& left, const Vec2d& right)
{
    return left.norm2() > right.norm2() ? left : right;
}
//*****************************************************************************
DivideAvgFunctor::DivideAvgFunctor(ConstPtr<uint32> nbElems)
: nbElems(nbElems)
{}
//*****************************************************************************
HOST DEVICE
Vec2d DivideAvgFunctor::operator()(const Vec2d& left, uint32 right)
{
    const uint32 denominator = nbElems[right];
    return (denominator==0) ? Vec2d(0.,0.) : left/double(denominator);
}
}
