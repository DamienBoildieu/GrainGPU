#include "film.cuh"
#include <opencv2/opencv.hpp>
#include "SPH/particles.cuh"
//#include <thrust/pair.h>

using namespace cv;

namespace film {
//*****************************************************************************
void Film::setAvgMass(float avgMass)
{
    for (auto& layer : mLayers) {
        layer.setAvgMass(avgMass);
    }
}
//*****************************************************************************
void Film::setMassStdev(float massStdev)
{
    for (auto& layer : mLayers) {
        layer.setMassStdev(massStdev);
    }
}
//*****************************************************************************
void Film::setRho0(float rho0)
{
    for (auto& layer : mLayers)
        layer.setRho0(rho0);
}
//*****************************************************************************
void Film::setMu(float mu)
{
    for (auto& layer : mLayers)
        layer.setMu(mu);
}
//*****************************************************************************
void Film::setElast(float elast)
{
    for (auto& layer : mLayers)
        layer.setElast(elast);
}
//*****************************************************************************
void Film::setFric(float fric)
{
    for (auto& layer : mLayers)
        layer.setFric(fric);
}
//*****************************************************************************
void Film::setSimuParams(const sph::SPHParameters& parameters)
{
    for (auto& layer : mLayers)
        layer.setSimuParams(parameters);
}
//*****************************************************************************
void Film::setWriteResults(bool writeResults)
{
    for (auto& layer : mLayers)
        layer.setWriteResults(writeResults);
}
//*****************************************************************************
void Film::setWriteForceStats(bool forceStats)
{
    for (auto& layer : mLayers)
        layer.setWriteForceStats(forceStats);
}
//*****************************************************************************
void Film::update(float dt, uint32 nbIte)
{
    for (uint32 i=0; i<mLayers.size(); i++)
        mLayers[i].update(dt, nbIte);
}
//*****************************************************************************
void Film::update(uint32 startX, uint32 startY, uint32 nbRows, uint32 nbCols,
    float dt, uint32 nbIte)
{
    for (uint32 i=0; i<mLayers.size(); i++)
        mLayers[i].update({startX, startY}, {nbCols, nbRows}, dt, nbIte);
}
//*****************************************************************************
void Film::computeFinalImage(uint32 rows, uint32 cols, uint32 xNbSamples,
    uint32 yNbSamples, bool debug)
{
    if (mLayers.size()<1)
        return;
    if (rows == 0)
        rows = mLayers[0].height();
    if (cols == 0)
        cols = mLayers[0].width();
    for (uint32 i=0; i<mLayers.size(); i++) {
        mLayers[i].convoluate(rows, cols, {xNbSamples, yNbSamples}, debug);
    }
}
//*****************************************************************************
void Film::computeFinalImage(uint32 startX, uint32 startY, uint32 nbRows, uint32 nbCols,
    uint32 rows, uint32 cols, uint32 xNbSamples, uint32 yNbSamples, bool debug)
{
    if (mLayers.size()<1)
        return;
    if (rows == 0)
        rows = mLayers[0].height();
    if (cols == 0)
        cols = mLayers[0].width();
    for (uint32 i=0; i<mLayers.size(); i++) {
        mLayers[i].convoluate({startX, startY}, {nbCols, nbRows}, rows, cols,
            {xNbSamples, yNbSamples}, debug);
    }
}
//*****************************************************************************
void Film::writeImage(const std::string& fileName) const
{
    if (mLayers.size()<1)
        return;
    const uint32 width = mLayers[0].width();
    const uint32 height = mLayers[0].height();
    int32 type = CV_8UC1;
    Mat image{(int32)height, (int32)width, type};
    for (uint32 i=0; i<mLayers.size(); i++) {
        auto& pixels = mLayers[i].hostPixels();
        for (uint32 j=0; j<height; j++)
            for (uint32 k=0; k<width; k++) {
                    image.at<uchar>(j,k) = pixels.intensities[j*width+k]*5.*255.;
                    
                    //Color case TODO
                    /*auto& pix = image.at<Vec3<uint32>>(j,k);
                    pix[i] = intensity[j*layers[i].getWidth()+k];*/
            }
    }
    imwrite(fileName, image);
}
//*****************************************************************************
const std::vector<Layer>& Film::layers() const
{
    return mLayers;
}
//*****************************************************************************
std::vector<Layer>& Film::layers()
{
    return mLayers;
}
}
