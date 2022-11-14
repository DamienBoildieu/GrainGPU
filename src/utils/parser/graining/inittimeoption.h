#pragma once
#include "grainingoption.hpp"

namespace parser {
class InitTimeOption : public GrainingOption
{
public:
    InitTimeOption(Graining& graining);
    InitTimeOption(const InitTimeOption& other) = default;
    InitTimeOption(InitTimeOption&& other) = default;
    virtual ~InitTimeOption() = default;
    InitTimeOption& operator=(const InitTimeOption& other) = default;
    InitTimeOption& operator=(InitTimeOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}