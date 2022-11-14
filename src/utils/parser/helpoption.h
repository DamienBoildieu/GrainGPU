#pragma once
#include "option.hpp"
#include <vector>
#include <memory>

namespace parser {
class HelpOption : public Option
{
public:
    HelpOption(const std::string& program, const std::vector<std::unique_ptr<Option>>& options);
    HelpOption(const HelpOption& other) = default;
    HelpOption(HelpOption&& other) = default;
    virtual ~HelpOption() = default;
    HelpOption& operator=(const HelpOption& other) = default;
    HelpOption& operator=(HelpOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    using Option::operator();
    void operator()(const std::vector<std::string>& args) override;

private:
    std::string program;
    const std::vector<std::unique_ptr<Option>>& options;
};
}
