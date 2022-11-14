#include "grainingController.cuh"
#include "utils/math.cuh"
#include <QTextStream>

//*****************************************************************************
sph::SPHParameters SimuParameters::toSPHParameters() const
{
    return {avgMass, massStdev, minMass, maxMass, rho0, mu, elast, fric};
}
//*****************************************************************************
GrainingController::GrainingController()
: mPixmap(), mFilm(), mHeight(), mWidth(), mInitialized(false),
  mOriginalImage(), mOriginalWidth(), mOriginalHeight(), mCroped(false)
{}
//*****************************************************************************
bool GrainingController::loadImage(const QString& path)
{
    if (!mPixmap.load(path)) {
        return false;
    }
    mHeight = mPixmap.height();
    mOriginalHeight = mHeight;
    mWidth = mPixmap.width();
    mOriginalWidth = mWidth;
    mDisplayPos = {0,0};
    mDisplaySize = {mWidth, mHeight};
    mCroped = false;
    return true;
}
//*****************************************************************************
bool GrainingController::init(bool isRgb, const SimuParameters& parameters,
    bool writeResults, bool writeForceStats)
{
    QImage img = mPixmap.toImage();
    int32 nbElems = mWidth*mHeight;
    uint32 nbLayers = isRgb ? 3 : 1;
    if (nbElems==0)
        return false;
    mOriginalImage.resize(nbLayers*nbElems);

    for (int32 y=0; y<mHeight; y++) {
        const QRgb *rowData = (QRgb*)img.constScanLine(y);
        for (int32 x=0; x<mWidth; x++) {
            QRgb color = rowData[x];
            if (isRgb) {
                mOriginalImage[y*mWidth+x] = qRed(color)/255.;
                mOriginalImage[(mWidth*mHeight)+y*mWidth+x] = qGreen(color)/255.;
                mOriginalImage[(2*mWidth*mHeight)+y*mWidth+x] = qBlue(color)/255.;
            } else
                mOriginalImage[y*mWidth+x] = qGray(color)/255.;
        }
    }
    auto start = std::chrono::system_clock::now();
    mFilm = {mWidth, mHeight, nbLayers, mOriginalImage, parameters.toSPHParameters(), writeResults,
        writeForceStats};
    auto end = std::chrono::system_clock::now();
    QTextStream out(stdout);
    out << "Init simulations : " << QString::number(std::chrono::duration<double>(end-start).count()) << Qt::endl;
    mInitialized = true;
    return true;
}
//*****************************************************************************
void GrainingController::update(uint32 nbIte, float dt)
{
    if (!mInitialized)
        return;
    auto start = std::chrono::system_clock::now();
    if (!mCroped)
        mFilm.update(dt, nbIte);
    else
        mFilm.update(mCropPos.first, mCropPos.second, mCropSize.second, mCropSize.first,
            dt, nbIte);
    auto end = std::chrono::system_clock::now();
    QTextStream out(stdout);
    out << "Updates " << nbIte << " times : " << QString::number(std::chrono::duration<double>(end-start).count()) << Qt::endl;
}
//*****************************************************************************
void GrainingController::computeImage(const QPair<uint32,uint32>& nbSamples,
    bool debug)
{
    if (!mInitialized)
        return;
    QTextStream out(stdout);
    out << "convoluate" << Qt::endl;

    auto start = std::chrono::system_clock::now();
    if (!mCroped) {
        mFilm.computeFinalImage(mHeight, mWidth, nbSamples.first, nbSamples.second, debug);
        mDisplayPos = {0 , 0};
        mDisplaySize = {mOriginalWidth, mOriginalHeight};
    } else {
        if (mCropSize.second>mCropSize.first) {
            mWidth = mHeight*mCropSize.first/mCropSize.second;
        } else {
            mHeight = mWidth*mCropSize.second/mCropSize.first;
        }
        mFilm.computeFinalImage(mCropPos.first, mCropPos.second, mCropSize.second, mCropSize.first,
            mHeight, mWidth, nbSamples.first, nbSamples.second, debug);
        mDisplayPos = mCropPos;
        mDisplaySize = mCropSize;
    }
    auto end = std::chrono::system_clock::now();
    out << "compute grain : " << QString::number(std::chrono::duration<double>(end-start).count()) << Qt::endl;
    out << "Graining finished" << Qt::endl;
    QImage img{(int32)mWidth, (int32)mHeight, QImage::Format_RGB32};
    auto& layers = mFilm.layers();
    for (int32 y=0; y<mHeight; y++) {
        QRgb *rowData = (QRgb*)img.scanLine(y);
        for (int32 x=0; x<mWidth; x++) {
            if (layers.size()==1) {
                float intensity = layers[0].hostPixels().intensities[y*mWidth+x];
                if (debug) {
                    QVector<uint32> decomposed = utils::naiveIntFactorization<QVector>(intensity);
                    float totalSamples = nbSamples.first*nbSamples.second;
                    //std::cerr << std::to_string(intensity) << std::endl;
                    for (auto& elem : decomposed)
                        elem = elem/totalSamples*255;
                    QVector<float> color(3u);
                    for (uint32 i=0u; i<3u; i++) {
                        color[i] = decomposed.size()<(i+1u) ? 0 : decomposed[i]/totalSamples;
                        //std::cerr << color[i] << std::endl << "=======================" << std::endl;
                    }
                    rowData[x] = QColor{uchar(color[0]*255), uchar(color[1]*255), uchar(color[2]*255)}.rgb();
                } else {
                    intensity *= 255;
                    rowData[x] = QColor{uchar(intensity), uchar(intensity), uchar(intensity)}.rgb();
                }
                
            } else {
                uchar red = layers[0].hostPixels().intensities[y*mWidth+x]*255;
                uchar green = layers[1].hostPixels().intensities[y*mWidth+x]*255;
                uchar blue = layers[2].hostPixels().intensities[y*mWidth+x]*255;
                rowData[x] = QColor{red, green, blue}.rgb();
            }
        }
    }
    mPixmap.convertFromImage(img);
}
//*****************************************************************************
void GrainingController::saveImage(const QString& filePath) const
{
    mPixmap.save(filePath);
}
//*****************************************************************************
void GrainingController::reset(const SimuParameters& parameters, bool writeResults,
    bool writeForceStats)
{
    if (!mInitialized)
        return;
    mFilm = {mOriginalWidth, mOriginalHeight, uint32(mFilm.layers().size()), mOriginalImage,
        parameters.toSPHParameters(), writeResults, writeForceStats};
}
//*****************************************************************************
void GrainingController::computeOriginalImage()
{
    if (!mInitialized)
        return;
    QImage img{int32(mOriginalWidth), int32(mOriginalHeight), QImage::Format_RGB32};
    const uint64 nbPixels = mOriginalHeight*mOriginalWidth;
    for (int32 y=0; y<mOriginalHeight; y++) {
        QRgb *rowData = (QRgb*)img.scanLine(y);
        for (int32 x=0; x<mOriginalWidth; x++) {
            if (mFilm.layers().size()==1ul) {
                uchar intensity = mOriginalImage[y*mOriginalWidth+x]*255;
                rowData[x] = QColor{intensity, intensity, intensity}.rgb();
            } else {
                uchar red = mOriginalImage[y*mOriginalWidth+x]*255;
                uchar green = mOriginalImage[nbPixels+y*mOriginalWidth+x]*255;
                uchar blue = mOriginalImage[2*nbPixels+y*mOriginalWidth+x]*255;
                rowData[x] = QColor{red, green, blue}.rgb();
            }
        }
    }
    mWidth = mOriginalWidth;
    mHeight = mOriginalHeight;
    mDisplayPos = {0,0};
    mDisplaySize = {mOriginalWidth, mOriginalHeight};
    mCroped = false;
    mPixmap.convertFromImage(img);
}
//*****************************************************************************
void GrainingController::setAvgMass(float avgMass)
{
    if (!mInitialized)
        return;
    mFilm.setAvgMass(avgMass);
}
//*****************************************************************************
void GrainingController::setMassStdev(float massStdev)
{
    if (!mInitialized)
        return;
    mFilm.setMassStdev(massStdev);
}
//*****************************************************************************
void GrainingController::setRho0(float rho0)
{
    if (!mInitialized)
        return;
    mFilm.setRho0(rho0);
}
//*****************************************************************************
void GrainingController::setMu(float mu)
{
    if (!mInitialized)
        return;
    mFilm.setMu(mu);
}
//*****************************************************************************
void GrainingController::setElast(float elast)
{
    if (!mInitialized)
        return;
    mFilm.setElast(elast);
}
//*****************************************************************************
void GrainingController::setFric(float fric)
{
    if (!mInitialized)
        return;
    mFilm.setFric(fric);
}
//*****************************************************************************
void GrainingController::setSimuParams(const SimuParameters& parameters)
{
    if (!mInitialized)
        return;
    mFilm.setSimuParams(parameters.toSPHParameters());
}
//*****************************************************************************
float GrainingController::computeRadius(float mass, float rho0) const
{
    return sph::radius(mass, rho0);
}
//*****************************************************************************
void GrainingController::setWriteResults(bool writeResults)
{
    mFilm.setWriteResults(writeResults);
}
//*****************************************************************************
void GrainingController::setWriteForceStats(bool forceStats)
{
    mFilm.setWriteForceStats(forceStats);
}
//*****************************************************************************
const QPixmap& GrainingController::pixmap() const
{
    return mPixmap;
}
//*****************************************************************************
uint32 GrainingController::width() const
{
    return mWidth;
}
//*****************************************************************************
void GrainingController::setWidth(uint32 width)
{
    mWidth = width;
}
//*****************************************************************************
uint32 GrainingController::height() const
{
    return mHeight;
}
//*****************************************************************************
void GrainingController::setHeight(uint32 height)
{
    mHeight = height;
}
//*****************************************************************************
uint32 GrainingController::originalWidth() const
{
    return mOriginalWidth;
}
//*****************************************************************************
uint32 GrainingController::originalHeight() const
{
    return mOriginalHeight;
}
//*****************************************************************************
bool GrainingController::initialized() const
{
     return mInitialized;
}
//*****************************************************************************
void GrainingController::resetInitialize()
{
    mInitialized = false;
}
//*****************************************************************************
void GrainingController::setFirstCorner(const QPointF& firstCorner)
{
    float ratioX = (mWidth/(float)mDisplaySize.first);
    float ratioY = (mHeight/(float)mDisplaySize.second);
    mCropPos.first = firstCorner.x()/ratioX + mDisplayPos.first;
    mCropPos.second = firstCorner.y()/ratioY + mDisplayPos.second;
    std::cerr << "x : " << mCropPos.first << std::endl;
    std::cerr << "y : " << mCropPos.second << std::endl;
}
//*****************************************************************************
void GrainingController::setSecondCorner(const QPointF& secondCorner)
{
    float ratioX = (mWidth/(float)mDisplaySize.first);
    float ratioY = (mHeight/(float)mDisplaySize.second);
    uint32 firstPointX = mCropPos.first;
    uint32 firstPointY = mCropPos.second;
    uint32 secondPointX = secondCorner.x()/ratioX + mDisplayPos.first;
    uint32 secondPointY = secondCorner.y()/ratioY + mDisplayPos.second;
    if (secondPointX < firstPointX)
        mCropPos.first = secondPointX;
    if (secondPointY < firstPointY)
        mCropPos.second = secondPointY;

    std::cerr << "x : " << mCropPos.first << std::endl;
    std::cerr << "y : " << mCropPos.second << std::endl;
    int32 width = secondPointX-firstPointX;
    std::cerr << "width : " << width << std::endl;
    mCropSize.first = width < 0 ? -width : width;
    int32 height = secondPointY-firstPointY;
    std::cerr << "height : " << height << std::endl;
    std::cerr << "---------------------------" << std::endl;
    mCropSize.second = height < 0 ? -height : height;
    mCropSize.first++;
    mCropSize.second++;
}
//*****************************************************************************
bool GrainingController::croped() const
{
    return mCroped;
}
//*****************************************************************************
void GrainingController::setCroped(bool croped)
{
    mCroped = croped;
}
