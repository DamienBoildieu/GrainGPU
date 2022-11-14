#pragma once
#include "utils/parser/option.hpp"
#include "IHM/graining.h"

namespace parser {
class GrainingOption : public Option
{
public:
    inline GrainingOption(const std::string& name, const std::string& alias, const std::string& message, Graining& graining);
    GrainingOption(const GrainingOption& other) = default;
    GrainingOption(GrainingOption&& other) = default;
    virtual ~GrainingOption() = default;
    GrainingOption& operator=(const GrainingOption& other) = default;
    GrainingOption& operator=(GrainingOption&& other) = default;

protected:
    inline Graining& getGraining();

private:
    Graining& graining;
};

//*****************************************************************************
//Definition
//*****************************************************************************
//*****************************************************************************
GrainingOption::GrainingOption(const std::string& name, const std::string& alias, const std::string& message, Graining& graining)
    : Option(name, alias, message), graining(graining)
{}
//*****************************************************************************
Graining& GrainingOption::getGraining()
{
    return graining;
}
}
