#include "inittimeoption.h"

namespace parser {
//*****************************************************************************
InitTimeOption::InitTimeOption(Graining& graining)
    : GrainingOption("--initTime", "-it", "write the init duration", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> InitTimeOption::copy() const
{
    return std::unique_ptr<Option>(new InitTimeOption(*this));
}
//*****************************************************************************
void InitTimeOption::operator()(const std::vector<std::string>& args)
{
    if (args.empty()) {
        auto& graining = getGraining();
        graining.setWriteInitTime(true);
    } else
        std::cout << getName() << " option doesn't take arguments" << std::endl;
}
}