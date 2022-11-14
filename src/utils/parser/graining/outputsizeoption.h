#pragma once
#include "grainingoption.hpp"

namespace parser {
class OutputSizeOption : public GrainingOption
{
public:
    OutputSizeOption(Graining& graining);
    OutputSizeOption(const OutputSizeOption& other) = default;
    OutputSizeOption(OutputSizeOption&& other) = default;
    virtual ~OutputSizeOption() = default;
    OutputSizeOption& operator=(const OutputSizeOption& other) = default;
    OutputSizeOption& operator=(OutputSizeOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}