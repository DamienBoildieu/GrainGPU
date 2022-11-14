#pragma once
#include <string>
#include <vector>
#include <iostream>
#include "memory.cuh"
#ifdef WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace utils {
constexpr char logDir[] = "log";
constexpr char rho[] = "rho";
constexpr char press[] = "press";
constexpr char force[] = "force";
constexpr char vel[] = "vel";
constexpr char beforePos[] = "beforePos";
constexpr char pos[] = "pos";
constexpr char neigh[] = "neigh";
constexpr char mass[] = "mass";
constexpr char radius[] = "radius";
constexpr char dirSep[] = "/";

const std::string rhoDir{std::string{logDir}+std::string{dirSep}+std::string{rho}};
const std::string pressDir{std::string{logDir}+std::string{dirSep}+std::string{press}};
const std::string forceDir{std::string{logDir}+std::string{dirSep}+std::string{force}};
const std::string velDir{std::string{logDir}+std::string{dirSep}+std::string{vel}};
const std::string posDir{std::string{logDir}+std::string{dirSep}+std::string{pos}};
const std::string beforePosDir{std::string{logDir}+std::string{dirSep}+std::string{beforePos}};
const std::string neighDir{std::string{logDir}+std::string{dirSep}+std::string{neigh}};
const std::string massDir{std::string{logDir}+std::string{dirSep}+std::string{mass}};
const std::string radiusDir{std::string{logDir}+std::string{dirSep}+std::string{radius}};

inline bool createDir(const std::string& name);

template<template<typename, typename...> typename Array, typename T, typename... Args>
std::ostream& writeArray(std::ostream& stream, const Array<T,Args...>& array);
template<template<typename, typename...> typename Array, typename T, typename... Args>
void writeArray(const std::string& filePath, const Array<T,Args...>& array);
template<typename T>
void writeDeviceArray(const std::string& filePath, ConstPtr<T> device, uint32 nbElems);
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace utils {
//*****************************************************************************
bool createDir(const std::string& name)
{
#ifdef WIN32
   _mkdir(name.c_str());//, S_IRWXU | S_IRWXG | S_IRWXO);
#else
   mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
#endif
   return true;
}
//*****************************************************************************
template<template<typename, typename...> typename Array, typename T, typename... Args>
std::ostream& writeArray(std::ostream& stream, const Array<T,Args...>& array)
{
    for (auto& elem : array)
        stream << elem << std::endl;
    return stream;
    
}
//*****************************************************************************
template<template<typename, typename...> typename Array, typename T, typename... Args>
void writeArray(const std::string& filePath, const Array<T,Args...>& array)
{
    std::ofstream file{filePath};
    writeArray(file, array);
}
//*****************************************************************************
template<typename T>
void writeDeviceArray(const std::string& filePath, ConstPtr<T> device, uint32 nbElems)
{
    std::ofstream file{filePath};
    std::vector<T> hArray(nbElems);
    cudaDeviceToHost(device, hArray.data(), nbElems);
    writeArray(file, hArray);
}
}
