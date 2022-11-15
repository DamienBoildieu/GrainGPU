#pragma once
#include "layer.cuh"
#include "utils/vec.cuh"
#include <vector>
#include <string>

//*****************************************************************************
//Declaration
//****************************************************************************
namespace film {
class Film {
public:
    Film() = default;
    Film(const Film& other) = default;
    template <template<typename, typename...> typename Array, typename... Args>
    Film(uint32 width, uint32 height, uint32 nbLayers, const Array<float>& intensity,
        const sph::SPHParameters& params, bool writeResults=false, bool writeForceStats=false);
    ~Film() = default;
    Film& operator=(const Film& right) = default;
    
    void setAvgMass(float avgMass);
    void setMassStdev(float massStdev);
    void setRho0(float rho0);
    void setMu(float mu);
    void setElast(float elast);
    void setFric(float fric);
    void setSimuParams(const sph::SPHParameters& parameters);

    void setWriteResults(bool writeResults);
    void setWriteForceStats(bool forceStats);

    void update(float dt, uint32 nbIte=1u);
    void update(uint32 startX, uint32 startY, uint32 nbRows, uint32 nbCols, float dt, uint32 nbIte);
    void computeFinalImage(uint32 rows=0, uint32 cols=0, uint32 xNbSamples=16u,
        uint32 yNbSamples=16u, bool debug = false);
    void computeFinalImage(uint32 startX, uint32 startY, uint32 nbRows, uint32 nbCols,
        uint32 rows, uint32 cols, uint32 xNbSamples, uint32 yNbSamples, bool debug = false);
    void writeImage(const std::string& fileName) const;
    const std::vector<Layer>& layers() const;
    std::vector<Layer>& layers();
private:
    std::vector<Layer> mLayers;
};
}
//*****************************************************************************
//Definition
//*****************************************************************************
namespace film {
//*****************************************************************************
template <template<typename, typename...> typename Array, typename... Args>
Film::Film(uint32 width, uint32 height, uint32 nbLayers, const Array<float>& intensity,
    const sph::SPHParameters& params, bool writeResults, bool writeForceStats)
: mLayers()
{
    float intensityCoeff = 0.2f;
    uint32 nbElems = intensity.size()/nbLayers;
    for (uint32 i=0; i<nbLayers; i++) {
        std::cout << "Layer " << i << std::endl;
        mLayers.push_back({width, height, intensityCoeff, intensity.begin()+i*nbElems,
            params, writeResults, writeForceStats});
    }
}
}
