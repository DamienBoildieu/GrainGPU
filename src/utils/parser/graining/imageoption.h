#pragma once
#include "grainingoption.hpp"

namespace parser {
class ImageOption : public GrainingOption
{
public:
    ImageOption(Graining& graining);
    ImageOption(const ImageOption& other) = default;
    ImageOption(ImageOption&& other) = default;
    virtual ~ImageOption() = default;
    ImageOption& operator=(const ImageOption& other) = default;
    ImageOption& operator=(ImageOption&& other) = default;
    std::unique_ptr<Option> copy() const override;

    void operator()(const std::vector<std::string>& args) override;
};
}
