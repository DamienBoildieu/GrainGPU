#include "outputsizeoption.h"

namespace parser {
//*****************************************************************************
OutputSizeOption::OutputSizeOption(Graining& graining)
    : GrainingOption("--outputSize", "-os", "set the output image size", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> OutputSizeOption::copy() const
{
    return std::unique_ptr<Option>(new OutputSizeOption(*this));
}
//*****************************************************************************
void OutputSizeOption::operator()(const std::vector<std::string>& args)
{
    if (args.size() == 2) {
        auto& graining = getGraining();
        graining.setOutputWidth(std::stoi(args[0]));
        graining.setOutputHeight(std::stoi(args[1]));
    } else
        std::cout << getName() << " option take two argument" << std::endl;
}
}
