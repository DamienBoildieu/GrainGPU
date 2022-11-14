#include "stats.cuh"
#include "functors.cuh"
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace utils {
//*****************************************************************************
DUniqueArrayPtr<Vec2d> mins(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys,
    uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput)
{
//    DPtr<const Vec2d> tForce(force);
//    DPtr<const uint32> tKeys(inputKeys);
    DUniqueArrayPtr<Vec2d> forceMin{cudaAllocator<Vec2d>(nbOutput)};
//    DPtr<Vec2d> tForceMin(forceMin.get());
//    cudaFill(forceMin.get(), 0u, nbOutput);
//    DPtr<uint32> tOutputKeys(outputKeys);
//    thrust::reduce_by_key(tKeys, tKeys+nbInput, tForce, tOutputKeys, tForceMin, thrust::equal_to<uint32>(),
//        MinFunctor());
    return forceMin;
}
//*****************************************************************************
DUniqueArrayPtr<Vec2d> maxs(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys,
    uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput)
{
//    DPtr<const Vec2d> tForce(force);
//    DPtr<const uint32> tKeys(inputKeys);
    DUniqueArrayPtr<Vec2d> forceMax{cudaAllocator<Vec2d>(nbOutput)};
//    DPtr<Vec2d> tForceMax(forceMax.get());
//    cudaFill(forceMax.get(), 0u, nbOutput);
//    DPtr<uint32> tOutputKeys(outputKeys);
//    thrust::reduce_by_key(tKeys, tKeys+nbInput, tForce, tOutputKeys, tForceMax, thrust::equal_to<uint32>(),
//        MaxFunctor());
    return forceMax;
}
//*****************************************************************************
DUniqueArrayPtr<Vec2d> avgs(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys,
    ConstPtr<uint32> elemsByInput, uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput)
{
//    DPtr<const Vec2d> tForce(force);
//    DPtr<const uint32> tKeys(inputKeys);
    DUniqueArrayPtr<Vec2d> forceAvg{cudaAllocator<Vec2d>(nbOutput)};
//    cudaFill(forceAvg.get(), 0u, nbOutput);
//    DPtr<Vec2d> tForceAvg(forceAvg.get());
//    DPtr<uint32> tOutputKeys(outputKeys);
//    DPtr<const uint32> tElemsByInput(elemsByInput);
//    auto lastKey = thrust::reduce_by_key(tKeys, tKeys+nbInput, tForce, tOutputKeys, tForceAvg, thrust::equal_to<uint32>(),
//                  thrust::plus<Vec2d>());
//    thrust::transform(tForceAvg, tForceAvg+(lastKey.first-tOutputKeys), tOutputKeys, tForceAvg,
//        DivideAvgFunctor{elemsByInput});
    return forceAvg;
}
}
