#pragma once
#include <type_traits>
#include "utils/cuda.cuh"
#include <cassert>
#include <iostream>

namespace utils {
template<typename T, uint64 N>
class Vec {
public:
    HOST DEVICE Vec(const T& val=T());
    HOST DEVICE Vec(T array[N]);
    HOST DEVICE Vec(const T& x, const T& y);
    HOST DEVICE Vec(const T& x, const T& y, const T& z);
    HOST DEVICE Vec(const T& x, const T& y, const T& z, const T& t);
    Vec(const Vec<T,N>& other) = default;
    Vec(Vec<T,N>&& other) = default;
    ~Vec() = default;
    Vec<T,N>& operator=(const Vec<T,N>& right) = default;
    Vec<T,N>& operator=(Vec<T,N>&& right) = default;

    HOST DEVICE Vec<T,N> operator+(const Vec<T,N>& right) const;
    HOST DEVICE Vec<T,N> operator-(const Vec<T,N>& right) const;
    HOST DEVICE Vec<T,N> operator*(const T& right) const;
    HOST DEVICE Vec<T,N> operator/(const T& right) const;
    template<typename U, uint64 M>
    HOST DEVICE inline friend Vec<U,M> operator*(const U& left, const Vec<U,M>& right);
    HOST DEVICE Vec<T,N> operator-() const;
    HOST DEVICE Vec<T,N>& operator+=(const Vec<T,N>& right);
    HOST DEVICE Vec<T,N>& operator-=(const Vec<T,N>& right);
    HOST DEVICE Vec<T,N>& operator*=(const T& right);
    HOST DEVICE Vec<T,N>& operator/=(const T& right);

    HOST DEVICE Vec<T,3> cross(const Vec<T,3>& right) const;
    HOST DEVICE T dot(const Vec<T,N>& right) const;
    
    HOST DEVICE T norm2() const;
    HOST DEVICE T norm() const;
    HOST DEVICE T normalize();

    HOST DEVICE const T& get(uint64 index) const;
    HOST DEVICE T& get(uint64 index);
    HOST DEVICE const T& x() const;
    HOST DEVICE const T& y() const;
    HOST DEVICE const T& z() const;
    HOST DEVICE const T& t() const;
    HOST DEVICE const T& w() const;
    HOST DEVICE T& x();
    HOST DEVICE T& y();
    HOST DEVICE T& z();
    HOST DEVICE T& t();
    HOST DEVICE T& w();

    HOST DEVICE const T& operator[](uint64 index) const;
    HOST DEVICE T& operator[](uint64 index);

    template<typename U, uint64 M>
    friend std::ostream& operator<<(std::ostream& stream, const Vec<U,M>& vec);

private:
    T elems[N];
};

template<typename T>
using Vec4 = Vec<T, 4>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;

template<typename T>
using Vec3 = Vec<T, 3>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

template<typename T>
using Vec2 = Vec<T, 2>;
using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
}
//*****************************************************************************
//Definitions
//*****************************************************************************
//*****************************************************************************
namespace utils {
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>::Vec(const T& val)
{
    for (uint64 i=0; i<N; i++)
        elems[i] = val;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>::Vec(T array[N])
{
    for (uint64 i=0; i<N; i++)
        elems[i] = array[i]; 
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>::Vec(const T& x, const T& y)
{
    static_assert(N==2);
    elems[0] = x;
    elems[1] = y;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>::Vec(const T& x, const T& y, const T& z)
{
    static_assert(N==3);
    elems[0] = x;
    elems[1] = y;
    elems[2] = z;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>::Vec(const T& x, const T& y, const T& z, const T& t)
{
    static_assert(N==4);
    elems[0] = x;
    elems[1] = y;
    elems[2] = z;
    elems[3] = t;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> Vec<T,N>::operator+(const Vec<T,N>& right) const
{
    Vec<T,N> res{*this};
    for (uint64 i=0; i<N; i++)
        res[i] += right[i];
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> Vec<T,N>::operator-(const Vec<T,N>& right) const
{
    Vec<T,N> res{*this};
    for (uint64 i=0; i<N; i++)
        res[i] -= right[i];
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> Vec<T,N>::operator*(const T& right) const
{
    Vec<T,N> res{*this};
    for (uint64 i=0; i<N; i++)
        res[i] *= right;
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> Vec<T,N>::operator/(const T& right) const
{
    assert(right!=0);
    Vec<T,N> res{*this};
    for (uint64 i=0; i<N; i++)
        res[i] /= right;
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> operator*(const T& left, const Vec<T,N>& right)
{
    Vec<T,N> res{left};
    for (uint64 i=0; i<N; i++)
        res[i] *= right[i];
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N> Vec<T,N>::operator-() const
{
    Vec<T,N> res{*this};
    for (uint64 i=0; i<N; i++)
        res[i] = - res[i];
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>& Vec<T,N>::operator+=(const Vec<T,N>& right)
{
    for (uint64 i=0; i<N; i++)
        elems[i] += right[i];
    return *this;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>& Vec<T,N>::operator-=(const Vec<T,N>& right)
{
    for (uint64 i=0; i<N; i++)
        elems[i] -= right[i];
    return *this;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>& Vec<T,N>::operator*=(const T& right)
{
    for (uint64 i=0; i<N; i++)
        elems[i] *= right;
    return *this;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,N>& Vec<T,N>::operator/=(const T& right)
{
    for (uint64 i=0; i<N; i++)
        elems[i] /= right;
    return *this;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE Vec<T,3> Vec<T,N>::cross(const Vec<T,3>& right) const
{
    static_assert(N==3);
    return {elems[1]*right[2] - elems[2]*right[1],
            elems[2]*right[0] - elems[0]*right[2],
            elems[0]*right[1] - elems[1]*right[0]};
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T Vec<T,N>::dot(const Vec<T,N>& right) const
{
    T res = 0;
    for (uint64 i=0; i<N; i++) {
        res += elems[i] * right[i];
    }
    return res; 
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T Vec<T,N>::norm2() const
{
    T res = 0;
    for (uint64 i=0; i<N; i++) {
        res += elems[i] * elems[i];
    }
    return res;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T Vec<T,N>::norm() const
{
    return sqrt(norm2());
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T Vec<T,N>::normalize()
{
    T norme = norm();
    if (norme==0)
        return norme;
    for (uint64 i=0; i<N; i++) {
        elems[i] /= norme;
    }
    return norme;
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::get(uint64 index) const
{
    return elems[index];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::get(uint64 index)
{
    return elems[index];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::x() const
{
    static_assert(N>0);
    return elems[0];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::y() const
{
    static_assert(N>1);
    return elems[1];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::z() const
{
    static_assert(N>2);
    return elems[2];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::t() const
{
    static_assert(N>3);
    return elems[3];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::w() const
{
    static_assert(N>3);
    return elems[3];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::x()
{
    static_assert(N>0);
    return elems[0];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::y()
{
    static_assert(N>1);
    return elems[1];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::z()
{
    static_assert(N>2);
    return elems[2];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::t()
{
    static_assert(N>3);
    return elems[3];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::w()
{
    static_assert(N>3);
    return elems[3];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE const T& Vec<T,N>::operator[](uint64 index) const
{
    return elems[index];
}
//*****************************************************************************
template<typename T, uint64 N>
HOST DEVICE T& Vec<T,N>::operator[](uint64 index)
{
    return elems[index];
}
//*****************************************************************************
template<typename T, uint64 N>
std::ostream& operator<<(std::ostream& stream, const Vec<T,N>& vec)
{
    for (uint32 i=0; i<N; i++) {
        stream << vec[i];
        if (i!=(N-1))
            stream << " ";
    }
    return stream;
}
}
