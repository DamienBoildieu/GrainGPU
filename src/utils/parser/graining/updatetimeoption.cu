#include "updatetimeoption.h"

namespace parser {
//*****************************************************************************
UpdateTimeOption::UpdateTimeOption(Graining& graining)
    : GrainingOption("--updateTime", "-ut", "write the update duration", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> UpdateTimeOption::copy() const
{
    return std::unique_ptr<Option>(new UpdateTimeOption(*this));
}
//*****************************************************************************
void UpdateTimeOption::operator()(const std::vector<std::string>& args)
{
    if (args.empty()) {
        auto& graining = getGraining();
        graining.setWriteUpdateTime(true);
    } else
        std::cout << getName() << " option doesn't take arguments" << std::endl;
}
}