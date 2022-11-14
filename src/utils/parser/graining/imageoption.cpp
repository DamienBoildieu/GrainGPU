#include "imageoption.h"

namespace parser {
//*****************************************************************************
ImageOption::ImageOption(Graining& graining)
    : GrainingOption("--image", "-i", "the input image path", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> ImageOption::copy() const
{
    return std::unique_ptr<Option>(new ImageOption(*this));
}
//*****************************************************************************
void ImageOption::operator()(const std::vector<std::string>& args)
{
    if (args.size() == 1) {
        auto& graining = getGraining();
        graining.setUseFile(true);
        graining.setInput(args[0]);
    } else
        std::cout << getName() << " option take one argument" << std::endl;
}
}
