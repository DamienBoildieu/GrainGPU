#pragma once
#include "grainingoption.hpp"

namespace parser {
class RandomOption : public GrainingOption
{
public:
    RandomOption(Graining& graining);
    RandomOption(const RandomOption& other) = default;
    RandomOption(RandomOption&& other) = default;
    virtual ~RandomOption() = default;
    RandomOption& operator=(const RandomOption& other) = default;
    RandomOption& operator=(RandomOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}