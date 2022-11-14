#include "graining.h"
#include <random>
#include <opencv2/opencv.hpp>
#include "utils/parser/graining/grainingoptions.h"
#include "utils/system.h"

//*****************************************************************************
Graining::Graining()
: film(), parser("Graining"), nbIte(1), dT(0.01), resSpatial(1.), width(), height(),
  input(), output("output.png"), outputWidth(0), outputHeight(0), useFile(false),
  useRandomImage(false), writeInitTime(false), writeUpdateTime(false), writeConvoTime(false)
{
    std::cerr << "Welcome to Graining!" << std::endl;
    parser.addOption(parser::ImageOption{*this});
    parser.addOption(parser::RandomOption{*this});
    parser.addOption(parser::IteOption{*this});
    parser.addOption(parser::InitTimeOption{*this});
    parser.addOption(parser::UpdateTimeOption{*this});
    parser.addOption(parser::ConvoTimeOption{*this});
    parser.addOption(parser::OutputSizeOption{*this});
}
//*****************************************************************************
film::Film& Graining::getFilm()
{
    return film;
}
//*****************************************************************************
bool Graining::getUseFile() const
{
    return useFile;
}
//*****************************************************************************
bool Graining::getUseRandomImage() const
{
    return useRandomImage;
}
//*****************************************************************************
void Graining::setNbIte(uint32 ite)
{
    nbIte = ite;
}
//*****************************************************************************
void Graining::setWidth(uint32 width)
{
    this->width = width;
}
//*****************************************************************************
void Graining::setHeight(uint32 height)
{
    this->height = height;
}
//*****************************************************************************
void Graining::setInput(const std::string& file)
{
    input = file;
}
//*****************************************************************************
void Graining::setOutputWidth(uint32 width)
{
    outputWidth = width;
}
//*****************************************************************************
void Graining::setOutputHeight(uint32 height)
{
    outputHeight = height;
}
//*****************************************************************************
void Graining::setUseFile(bool use)
{
    useFile = use;
}
//*****************************************************************************
void Graining::setUseRandomImage(bool use)
{
    useRandomImage = use;
}
//*****************************************************************************
void Graining::setWriteInitTime(bool use)
{
    writeInitTime = use;
}
//*****************************************************************************
void Graining::setWriteUpdateTime(bool use)
{
    writeUpdateTime = use;
}
//*****************************************************************************
void Graining::setWriteConvoTime(bool use)
{
    writeConvoTime = use;
}
//*****************************************************************************
void Graining::run()
{
    if (writeUpdateTime || writeConvoTime)
        utils::createDir("chrono");
    std::cerr << "Update" << std::endl;
    auto start = std::chrono::system_clock::now();
    film.update(dT, nbIte);
    auto end = std::chrono::system_clock::now();
    double avg = std::chrono::duration<double>(end-start).count()/(double)nbIte;
    uint32 nbParticles = film.layers()[0].SPH().hostWorlds().particles.pos.size();
    if (writeUpdateTime) {
        std::ofstream updateFile{"chrono/update.txt", std::ios_base::app};
        updateFile << width << " " << height << " " << nbParticles << " " << nbIte 
            << " " << avg << std::endl;
    }
    
    std::cerr << "Update duration avg : " << std::to_string(avg) << std::endl;
    
    std::cerr << "Convoluate" << std::endl;
    start = std::chrono::system_clock::now();
    film.computeFinalImage(outputWidth, outputHeight);
    end = std::chrono::system_clock::now();
    if (writeConvoTime) {
        std::ofstream convoluateFile{"chrono/convo.txt", std::ios_base::app};
        convoluateFile << width << " " << height << " " << nbParticles << " "
            << std::chrono::duration<double>(end-start).count() << std::endl;
    }

    std::cerr << "Image computation duration : "
        << std::to_string(std::chrono::duration<double>(end-start).count()) << std::endl;

    auto& layers = film.layers();
    std::cerr << "Write computed image" << std::endl;
    int32 type = CV_8UC1;
    cv::Mat image{(int32)outputHeight, (int32)outputWidth, type};
    for (uint32 j=0; j<height; j++) {
      for (uint32 i=0; i<width; i++) {
        image.at<uchar>(j,i) = layers[0].hostPixels().intensities[j*width+i]*255;
      }
    }
    cv::imwrite(output, image);
    std::cerr << "Image writed" << std::endl;

    std::cerr << "End" << std::endl;
}
//*****************************************************************************
std::vector<float> Graining::loadGrayImage()
{
    std::cerr << "loadImage" << std::endl;
    auto image_color = cv::imread(input);
    cv::Mat image;
    cvtColor(image_color, image, CV_BGR2GRAY);
    width = image.cols;
    height = image.rows;
    int32 nbElems = width*height;
    std::vector<float> intensities(nbElems);
    for (int32 x=0; x<height; x++) {
        for (int32 y=0; y<width; y++) {
            intensities[x*width+y] = image.at<uchar>(x,y)/255.;
        }
    }
    std::cerr << "Image loaded" << std::endl;
    return intensities;
}
//*****************************************************************************
std::vector<float> Graining::generateGrayImage()
{
    //std::cerr << "Image to generate" << std::endl;
    //std::vector<float> image(width*height);
    //std::default_random_engine generator;
    //std::uniform_real_distribution<float> distribution(0.,1.);
    //auto random = bind(distribution, generator);
    //for (uint32 i=0; i<image.size(); i++)
    //    image[i] = random();
    //std::cerr << "Image genrated" << std::endl;
    //return image;
    return {0.8};
}
//*****************************************************************************
void Graining::writeGrayImage(const std::vector<float>& intensities, const std::string& name)
{
    std::cerr << "Write Generated image" << std::endl;
    int32 type = CV_8UC1;
    cv::Mat image{(int32)height, (int32)width, type};
    for (uint32 j=0; j<height; j++) {
      for (uint32 i=0; i<width; i++) {
        image.at<uchar>(j,i) = intensities[j*width+i]*255;
      }
    }
    cv::imwrite(name, image);
    std::cerr << "Image writed" << std::endl;
}
