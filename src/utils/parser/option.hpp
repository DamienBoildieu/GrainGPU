#pragma once
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <memory>

//*****************************************************************************
//Declaration
//*****************************************************************************
namespace parser {
class Option
{
public:
    Option() = default;
    inline Option(const std::string& name, const std::string& alias, const std::string& message);
    Option(const Option& other) = default;
    Option(Option&& other) = default;
    virtual ~Option() = default;
    Option& operator=(const Option& other) = default;
    Option& operator=(Option&& other) = default;

    virtual std::unique_ptr<Option> copy() const = 0;
    inline const std::string& getName() const;
    inline const std::string& getAlias() const;
    inline const std::string& getMessage() const;

    virtual void operator()(const std::vector<std::string>&) = 0;
private:
    std::string name;
    std::string alias;
    std::string message;
};
//*****************************************************************************
//Definition
//*****************************************************************************
//*****************************************************************************
Option::Option(const std::string& name, const std::string& alias, const std::string& message)
    : name(name), alias(alias), message(message)
{}

//*****************************************************************************
const std::string& Option::getName() const
{
    return name;
}

//*****************************************************************************
const std::string& Option::getAlias() const
{
    return alias;
}
//*****************************************************************************
const std::string& Option::getMessage() const
{
    return message;
}
}
