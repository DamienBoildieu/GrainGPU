#include "helpoption.h"
#include <iostream>

namespace parser {
//*****************************************************************************
HelpOption::HelpOption(const std::string& program, const std::vector<std::unique_ptr<Option>>& options)
    : Option("--help", "-h", "print this help message"), program(program), options(options)
{}
//*****************************************************************************
std::unique_ptr<Option> HelpOption::copy() const
{
    return std::unique_ptr<Option>(new HelpOption(*this));
}
//*****************************************************************************
void HelpOption::operator()(const std::vector<std::string>& args)
{
    if (args.empty()) {
        std::cout << program << " help message" << std::endl
            << "===========================================================" << std:: endl
            << "Following options are available : " << std::endl;
        for (auto& option: options)
            std::cout << option->getName() << "(" << option->getAlias() << ")" << " : " << option->getMessage() << std::endl;
        exit(0);
    } else
        std::cout << getName() << " option doesn't take arguments" << std::endl;
}
}
