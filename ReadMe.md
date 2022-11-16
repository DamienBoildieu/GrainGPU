# GainGPU
A SPH based image graining application using CUDA for the SPH simulation and Qt for the GUI.

## Dependencies
**Note**: These are the developmental system specs. Others versions may work.

* Linux
    * The provided CMake file only works on Linux at the moment
* CUDA 11.8
* OpenCV 4.5
* Qt6

## Installation
Once dependencies have been installed:
```
cmake . -B build
cd build
make
./Grains
```
![GUI](docs/img/gui.png?raw=true "GUI screenshot")
