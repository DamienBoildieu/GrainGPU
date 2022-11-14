#pragma once
#include "cuda.cuh"

namespace utils {
//*****************************************************************************
DEVICE double normal01Pdf(double x);
//*****************************************************************************
DEVICE double normal01Cdf(double x);
//*****************************************************************************
DEVICE double normal01CdfInv(double cdf);
//*****************************************************************************
DEVICE double normalPdf(double x, double mu, double sigma);
//*****************************************************************************
DEVICE double normalCdf(double x, double mu, double sigma);
//*****************************************************************************
DEVICE double normalCdfInv(double cdf, double mu, double sigma);
//*****************************************************************************
DEVICE double normalTruncPdf(double x, double mu, double sigma, double min, double max);
//*****************************************************************************
DEVICE double normalTruncCdf(double x, double mu, double sigma, double min, double max);
//*****************************************************************************
DEVICE double normalTruncCdfInv(double cdf, double mu, double sigma, double min, double max);
//*****************************************************************************
DEVICE double normalTruncMinPdf(double x, double mu, double sigma, double min);
//*****************************************************************************
DEVICE double normalTruncMinCdf(double x, double mu, double sigma, double min);
//*****************************************************************************
DEVICE double normalTruncMinCdfInv(double cdf, double mu, double sigma, double min);
//*****************************************************************************
DEVICE double normalTruncMaxPdf(double x, double mu, double sigma, double max);
//*****************************************************************************
DEVICE double normalTruncMaxCdf(double x, double mu, double sigma, double max);
//*****************************************************************************
DEVICE double normalTruncMaxCdfInv(double cdf, double mu, double sigma, double max);
}