#pragma once
#include "Film/film.cuh"
#include "utils/parser/parser.h"
#include <chrono>
#include <fstream>
#include "utils/system.h"

class Graining {
public:
    Graining();
    Graining(const Graining& other) = default;
    Graining(Graining&& other) = default;
    virtual ~Graining() = default;
    Graining& operator=(const Graining& other) = default;
    Graining& operator=(Graining&& other) = default;

    film::Film& getFilm();
    bool getUseFile() const;
    bool getUseRandomImage() const;

    void setNbIte(uint32 ite);
    void setWidth(uint32 width);
    void setHeight(uint32 height);
    void setInput(const std::string& file);
    void setOutputWidth(uint32 width);
    void setOutputHeight(uint32 height);
    void setUseFile(bool use);
    void setUseRandomImage(bool use);
    void setWriteInitTime(bool use);
    void setWriteUpdateTime(bool use);
    void setWriteConvoTime(bool use);

    void run();
    template <template <typename, typename...> typename Array, typename ...Args>
    void run(const Array<std::string>& line);
private:
    std::vector<float> loadGrayImage();
    std::vector<float> generateGrayImage();
    void writeGrayImage(const std::vector<float>& intensities, const std::string& name);
    
    film::Film film;
    parser::Parser parser;
    uint32 nbIte;
    float dT;
    float resSpatial;
    uint32 width;
    uint32 height;
    std::string input;
    std::string output;
    uint32 outputWidth;
    uint32 outputHeight;
    bool useFile;
    bool useRandomImage;
    bool writeInitTime;
    bool writeUpdateTime;
    bool writeConvoTime;
};
//*****************************************************************************
//Definitions
//*****************************************************************************
//*****************************************************************************
template <template <typename, typename...> typename Array, typename ...Args>
void Graining::run(const Array<std::string>& line)
{
    parser.parse(line);
    if (useFile && useRandomImage) {
        std::cerr << "You have to chose between use an input file or generate a random image" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<float> intensities;
    if (useFile) {
        intensities = loadGrayImage();
    } else if (useRandomImage) {
        intensities = generateGrayImage();
        writeGrayImage(intensities, "input.png");
    }
    std::cerr << "Init" << std::endl;
    auto start = std::chrono::system_clock::now();
    bool debug = (width*height)==1;
    film = {width, height, resSpatial, 1, intensities, debug};
    auto end = std::chrono::system_clock::now();
    if (useFile) {
        film.writeImage("input.png");
    }
    uint32 nbParticles = film.layers()[0].SPH().hostWorlds().particles.pos.size();
    if (writeInitTime) {
        utils::createDir("chrono");
        std::ofstream initFile{"chrono/init.txt", std::ios_base::app};
        initFile << width << " " << height << " " << " " << nbParticles << " " 
            << std::chrono::duration<double>(end-start).count() << std::endl;
    }
    std::cerr << "Init duration : " << std::to_string(std::chrono::duration<double>(end-start).count()) 
        << std::endl;
    run();
}
