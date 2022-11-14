#include "utils/parser/parser.h"

namespace parser {
//*****************************************************************************
Parser::Parser(const std::string& program)
    : options()
{
    options.push_back(std::unique_ptr<Option>(new HelpOption(program, options)));
}
//*****************************************************************************
void Parser::addOption(const Option& option)
{
    options.push_back(option.copy());
}
}
