#include "iteoption.h"

namespace parser {
//*****************************************************************************
IteOption::IteOption(Graining& graining)
    : GrainingOption("--iterations", "-ite", "specify simulation iterations", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> IteOption::copy() const
{
    return std::unique_ptr<Option>(new IteOption(*this));
}
//*****************************************************************************
void IteOption::operator()(const std::vector<std::string>& args)
{
    if (args.size() == 1) {
        auto& graining = getGraining();
        graining.setNbIte(std::stof(args[0]));
    } else
        std::cout << getName() << " option take one argument" << std::endl;
}
}
