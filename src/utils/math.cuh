#pragma once
#include "define.h"

namespace utils {
template <template<typename, typename...> typename Array, typename... Args>
Array<uint32> naiveIntFactorization(uint32 number);
}

namespace utils {
template <template<typename, typename...> typename Array, typename... Args>
Array<uint32> naiveIntFactorization(uint32 number)
{
    uint32 divisor = 2u;
    Array<uint32> result;
    while (number > 1u) {
        result.push_back(0u);
        while (number%divisor==0u) {
            result[divisor-2u]++;
            number/=divisor;
        } 
        divisor++;
    }
    return result;
}
}