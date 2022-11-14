#include "normalDistribution.cuh"
#include "define.h"

namespace utils {
//*****************************************************************************
DEVICE
double normal01Pdf(double x)
{
    return 1./sqrt(pi2)*exp(-.5*(x*x));
}
//*****************************************************************************
DEVICE
double normal01Cdf(double x)
{
    return .5*(1+erf(x/(sqrt(2.))));
}
//*****************************************************************************
DEVICE
double normal01CdfInv(double cdf)
{
    return sqrt(2.)*erfinv(2.*cdf-1.);
}
//*****************************************************************************
DEVICE
double normalPdf(double x, double mu, double sigma)
{
    x = (x-mu)/sigma;
    return 1./(sigma*sqrt(pi2))*exp(-.5*(x*x));
}
//*****************************************************************************
DEVICE
double normalCdf(double x, double mu, double sigma)
{
    return normal01Cdf((x-mu)/sigma);
}
//*****************************************************************************
DEVICE
double normalCdfInv(double cdf, double mu, double sigma)
{
    return normal01CdfInv(cdf)*sigma+mu;
}
//*****************************************************************************
DEVICE
double normalTruncPdf(double x, double mu, double sigma, double min, double max)
{
    x = (x-mu)/sigma;
    const double a = (min-mu)/sigma;
    const double b = (max-mu)/sigma;
    return normal01Pdf(x)/(sigma*(normal01Cdf(b)-normal01Cdf(a)));
}
//*****************************************************************************
DEVICE
double normalTruncCdf(double x, double mu, double sigma, double min, double max)
{
    x = (x-mu)/sigma;
    const double a = (min-mu)/sigma;
    const double b = (max-mu)/sigma;
    const double cdfA = normal01Cdf(a);
    const double cdfB = normal01Cdf(b);
    return (normal01Cdf(x)-cdfA)/(cdfB-cdfA);

}
//*****************************************************************************
DEVICE
double normalTruncCdfInv(double cdf, double mu, double sigma, double min, double max)
{
    const double a = (min-mu)/sigma;
    const double b = (max-mu)/sigma;
    const double cdfA = normal01Cdf(a);
    const double cdfB = normal01Cdf(b);
    return mu+sigma*normal01CdfInv((cdfB-cdfA)*cdf+cdfA);
}
//*****************************************************************************
DEVICE
double normalTruncMinPdf(double x, double mu, double sigma, double min)
{
    x = (x-mu)/sigma;
    const double a = (min-mu)/sigma;
    return normal01Pdf(x)/(sigma*(1.-normal01Cdf(a)));
}
//*****************************************************************************
DEVICE
double normalTruncMinCdf(double x, double mu, double sigma, double min)
{
    x = (x-mu)/sigma;
    const double a = (min-mu)/sigma;
    const double cdfA = normal01Cdf(a);
    return (normal01Cdf(x)-cdfA)/(1.-cdfA);

}
//*****************************************************************************
DEVICE
double normalTruncMinCdfInv(double cdf, double mu, double sigma, double min)
{
    const double a = (min-mu)/sigma;
    const double cdfA = normal01Cdf(a);
    return mu+sigma*normal01CdfInv((1.-cdfA)*cdf+cdfA);
}
//*****************************************************************************
DEVICE
double normalTruncMaxPdf(double x, double mu, double sigma, double max)
{
    x = (x-mu)/sigma;
    const double b = (max-mu)/sigma;
    return normal01Pdf(x)/(sigma*normal01Cdf(b));
}
//*****************************************************************************
DEVICE
double normalTruncMaxCdf(double x, double mu, double sigma, double min, double max)
{
    x = (x-mu)/sigma;
    const double b = (max-mu)/sigma;
    const double cdfB = normal01Cdf(b);
    return normal01Cdf(x)/cdfB;
}
//*****************************************************************************
DEVICE
double normalTruncMaxCdfInv(double cdf, double mu, double sigma, double min, double max)
{
    const double b = (max-mu)/sigma;
    const double cdfB = normal01Cdf(b);
    return mu+sigma*normal01CdfInv(cdfB*cdf);
}
}