#pragma once
#include "grainingoption.hpp"

namespace parser {
class ConvoTimeOption : public GrainingOption
{
public:
    ConvoTimeOption(Graining& graining);
    ConvoTimeOption(const ConvoTimeOption& other) = default;
    ConvoTimeOption(ConvoTimeOption&& other) = default;
    virtual ~ConvoTimeOption() = default;
    ConvoTimeOption& operator=(const ConvoTimeOption& other) = default;
    ConvoTimeOption& operator=(ConvoTimeOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}