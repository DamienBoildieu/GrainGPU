#pragma once
#include "Film/film.cuh"
#include <QVector>
#include <QPixmap>
#include <QPair>

//*****************************************************************************
struct SimuParameters {
    float avgMass;
    float massStdev;
    float minMass;
    float maxMass;
    float rho0;
    float mu;
    float elast;
    float fric;

    sph::SPHParameters toSPHParameters() const;
};
//*****************************************************************************
class GrainingController
{
public:
    GrainingController();
    GrainingController(const GrainingController& other) = default;
    GrainingController(GrainingController&& other) = default;
    ~GrainingController() = default;
    GrainingController& operator=(const GrainingController& right) = default;
    GrainingController& operator=(GrainingController&& right) = default;

    bool loadImage(const QString& path);
    bool init(bool isRgb, const SimuParameters& parameters, bool writeResults,
        bool writeForceStats);
    void update(uint32 nbIte, float dt);
    void computeImage(const QPair<uint32,uint32>& nbSamples, bool debug);
    void saveImage(const QString& filePath) const;
    void reset(const SimuParameters& parameters, bool writeResults, bool writeForceStats);
    void computeOriginalImage();

    void setAvgMass(float avgMass);
    void setMassStdev(float massStdev);
    void setRho0(float rho0);
    void setMu(float mu);
    void setElast(float elast);
    void setFric(float fric);
    void setSimuParams(const SimuParameters& parameters);
    float computeRadius(float mass, float rho0) const;

    void setWriteResults(bool writeResults);
    void setWriteForceStats(bool forceStats);

    const QPixmap& pixmap() const;

    uint32 width() const;
    void setWidth(uint32 newWidth);
    uint32 height() const;
    void setHeight(uint32 newHeight);
    uint32 originalWidth() const;
    uint32 originalHeight() const;
    bool initialized() const;
    void resetInitialize();

    void setFirstCorner(const QPointF& firstCorner);
    void setSecondCorner(const QPointF& secondCorner);
    bool croped() const;
    void setCroped(bool croped);
private:
    QPixmap mPixmap;
    
    film::Film mFilm;
    uint32 mHeight;
    uint32 mWidth;
    bool mInitialized;
    QVector<float> mOriginalImage;
    uint32 mOriginalWidth;
    uint32 mOriginalHeight;
    bool mCroped;
    QPair<uint32,uint32> mCropPos;
    QPair<uint32,uint32> mCropSize;
    QPair<uint32,uint32> mDisplayPos;
    QPair<uint32,uint32> mDisplaySize;
};
