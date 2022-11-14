#include "randomoption.h"

namespace parser {
//*****************************************************************************
RandomOption::RandomOption(Graining& graining)
    : GrainingOption("--random", "-r", "generate a random image size with specified width and height", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> RandomOption::copy() const
{
    return std::unique_ptr<Option>(new RandomOption(*this));
}
//*****************************************************************************
void RandomOption::operator()(const std::vector<std::string>& args)
{
    if (args.size() == 2) {
        auto& graining = getGraining();
        graining.setUseRandomImage(true);
        graining.setWidth(std::stoi(args[0]));
        graining.setHeight(std::stoi(args[1]));
    } else
        std::cout << getName() << " option take two argument" << std::endl;
}
}
