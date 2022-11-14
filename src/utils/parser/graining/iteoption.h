#pragma once
#include "grainingoption.hpp"

namespace parser {
class IteOption : public GrainingOption
{
public:
    IteOption(Graining& graining);
    IteOption(const IteOption& other) = default;
    IteOption(IteOption&& other) = default;
    virtual ~IteOption() = default;
    IteOption& operator=(const IteOption& other) = default;
    IteOption& operator=(IteOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}