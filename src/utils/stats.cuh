#pragma once
#include "memory.cuh"
#include "vec.cuh"

namespace utils {
DUniqueArrayPtr<Vec2d> mins(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys,
    uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput);

DUniqueArrayPtr<Vec2d> maxs(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys,
    uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput);

DUniqueArrayPtr<Vec2d> avgs(ConstPtr<Vec2d> force, ConstPtr<uint32> inputKeys, ConstPtr<uint32> elemsByInput,
    uint32 nbInput, Ptr<uint32> outputKeys, uint32 nbOutput);
}
