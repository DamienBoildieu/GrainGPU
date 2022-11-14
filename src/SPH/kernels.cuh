#pragma once
#include "utils/cuda.cuh"
#include "utils/vec.cuh"
#include "utils/memory.cuh"

namespace sph {
//*******************************************************************************
GLOBAL void getNeighbors(utils::Ptr<uint32> nbNeighbor, utils::ConstPtr<utils::Vec2f> pos,
    utils::ConstPtr<uint32> cellsStop, utils::ConstPtr<uint32> worldKey, utils::Vec2<uint32> dim,
    utils::ConstPtr<utils::Vec2f> lowerLimits, uint32 partNumber, uint32 worldBegin, float maxRadius);
//*****************************************************************************
GLOBAL void createGrid(utils::ConstPtr<utils::Vec2f> pos, utils::ConstPtr<uint32> worldKey,
    utils::Ptr<uint32> indexes, utils::Ptr<uint32> offset, utils::Ptr<uint32> partByCell,
    uint32 partNumber, uint32 worldBegin, utils::Vec2<uint32> dim, utils::ConstPtr<utils::Vec2f> lowerLimits,
    float maxRadius);
//*****************************************************************************
GLOBAL void computeRhoP(utils::ConstPtr<utils::Vec2f> pos, utils::ConstPtr<float> mass,
    utils::ConstPtr<float> radius, utils::Ptr<float> rho, utils::Ptr<float> p,
    utils::ConstPtr<uint32> cellsStop, utils::ConstPtr<uint32> worldKey, utils::Vec2<uint32> dim,
    utils::ConstPtr<utils::Vec2f> lowerLimits, uint32 partNumber, uint32 worldBegin,
    float maxRadius, float rho0, float taitsB);
//*****************************************************************************
GLOBAL void computeForces(utils::ConstPtr<utils::Vec2f> pos, utils::ConstPtr<float> mass,
    utils::ConstPtr<float> radius, utils::ConstPtr<float> rho, utils::ConstPtr<float> p,
    utils::Ptr<utils::Vec2d> force, utils::ConstPtr<utils::Vec2f> vel, 
    utils::ConstPtr<utils::Vec2d> externalForces, utils::Ptr<uint32> cellsStop,
    utils::ConstPtr<uint32> worldKey, utils::Vec2<uint32> dim, utils::ConstPtr<utils::Vec2f> lowerLimits,
    uint32 partNumber, uint32 worldBegin, float maxRadius, float mu, utils::Ptr<utils::Vec2d> fP,
    utils::Ptr<utils::Vec2d> fV, utils::Ptr<utils::Vec2d> fE);
//*****************************************************************************
GLOBAL void integrate(utils::Ptr<utils::Vec2f> pos, utils::ConstPtr<float> rho,
    utils::Ptr<utils::Vec2f> vel, utils::ConstPtr<utils::Vec2d> force, float dt, uint32 partNumber);
//*****************************************************************************
GLOBAL void collision(utils::Ptr<utils::Vec2f> pos, utils::Ptr<utils::Vec2f> vel,
    utils::ConstPtr<uint32> worldKey, utils::ConstPtr<utils::Vec2f> lowerLimits,
    utils::ConstPtr<utils::Vec2f> upperLimits, float elast, float fric, uint32 partNumber,
    uint32 worldBegin);
//*****************************************************************************
GLOBAL void updateMass(utils::Ptr<float> mass, utils::Ptr<float> radius,
    float avg, float dAvg, float dStdev, float rho0, uint32 partNumber);
}
