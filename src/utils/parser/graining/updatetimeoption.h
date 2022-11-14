#pragma once
#include "grainingoption.hpp"

namespace parser {
class UpdateTimeOption : public GrainingOption
{
public:
    UpdateTimeOption(Graining& graining);
    UpdateTimeOption(const UpdateTimeOption& other) = default;
    UpdateTimeOption(UpdateTimeOption&& other) = default;
    virtual ~UpdateTimeOption() = default;
    UpdateTimeOption& operator=(const UpdateTimeOption& other) = default;
    UpdateTimeOption& operator=(UpdateTimeOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}