#include "convotimeoption.h"

namespace parser {
//*****************************************************************************
ConvoTimeOption::ConvoTimeOption(Graining& graining)
    : GrainingOption("--convoTime", "-ct", "write the convoluate duration", graining)
{}
//*****************************************************************************
std::unique_ptr<Option> ConvoTimeOption::copy() const
{
    return std::unique_ptr<Option>(new ConvoTimeOption(*this));
}
//*****************************************************************************
void ConvoTimeOption::operator()(const std::vector<std::string>& args)
{
    if (args.empty()) {
        auto& graining = getGraining();
        graining.setWriteConvoTime(true);
    } else
        std::cout << getName() << " option doesn't take arguments" << std::endl;
}
}