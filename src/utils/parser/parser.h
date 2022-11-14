#pragma once
#include <vector>
#include <list>
#include <string>
#include <memory>
#include "option.hpp"
#include "helpoption.h"

//*****************************************************************************
//Declarations
//*****************************************************************************
namespace parser {
class Parser
{
public:
    Parser(const std::string& program);
    Parser(const Parser& other) = default;
    Parser(Parser&& other) = default;
    virtual ~Parser() = default;
    Parser& operator=(const Parser& other) = default;
    Parser& operator=(Parser&& other) = default;

    void addOption(const Option& option);

    template <template <typename, typename...> typename Array, typename ...Args>
    void parse(const Array<std::string>& line);

private:
    std::vector<std::unique_ptr<Option>> options;
};
}
//*****************************************************************************
//Definitions
//*****************************************************************************
namespace parser {
//*****************************************************************************
template <template <typename, typename...> typename Array, typename ...Args>
void Parser::parse(const Array<std::string>& line)
{
    auto ite = line.begin();
    std::vector<std::string> args;
    Option* opt = nullptr;
    for (; ite!=line.end(); ite++) {
        Option* nextOpt = nullptr;
        for (auto& option : options) {
            if (*ite==option->getName() || *ite==option->getAlias()) {
                nextOpt = option.get();
                break;
            }
        }
        if (nextOpt) {
            if (opt) {
                (*opt)(args);
                args.clear();
            }
            opt = nextOpt;
        } else {
            if (opt)
                args.push_back(*ite);
        }
    }
    if (opt)
        (*opt)(args);
}
}