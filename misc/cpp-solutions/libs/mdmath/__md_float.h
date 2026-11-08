#pragma once

#include <concepts>

#include "./typedefs.h"

namespace md::math
{

    template <typename float_t = float32, int _DefW = 0>
    struct alignas(sizeof(float_t) * 4) __md_float4
    {
        static_assert(std::is_same_v<float_t, float32> || std::is_same_v<float_t, float64>,
        "Only float32 or float64 are allowed");

        float_t x{0};
        float_t y{0};
        float_t z{0};
        float_t w{static_cast<float_t>(_DefW)};

        __md_float4() = default;
        __md_float4(const __md_float4& p)
        {
            x = p.x; y = p.y;
            z = p.z; w = p.w;
        }
        __md_float4(__md_float4&& p)
        {
            this.x = p.x; this.y = p.y;
            this.z = p.z; this.w = p.w;
            p.x = 0; p.y = 0; p.z = 0; p.w = 0;
        }
        __md_float4(float_t _x, float_t _y, float_t _z = 0, float_t _w = 0)
        {
            this->x = _x; this->y = _y;
            this->z = _z; this->w = _w;
        }
        __md_float4(float_t arr[4]) : x(arr[0]), y(arr[1]), z(arr[2]), w(arr[3]) {}
        __md_float4 operator+(const __md_float4& p) const
        {
            return __md_float4(x + p.x, y + p.y, z + p.z, w + p.w);
        }
        __md_float4& operator+=(const __md_float4& p)
        {
            x += p.x;
            y += p.y;
            z += p.z;
            w += p.w;
            return *this;
        }
        __md_float4 operator-(const __md_float4& p) const
        {
            return __md_float4(x - p.x, y - p.y, z - p.z, w - p.w);
        }
        __md_float4& operator-=(const __md_float4& p)
        {
            x -= p.x;
            y -= p.y;
            z -= p.z;
            w -= p.w;
            return *this;
        }
        __md_float4& operator=(const __md_float4& p)
        {
            if (this == &p) { return *this; }
            x = p.x;
            y = p.y;
            z = p.z;
            w = p.w;
            return *this;
        }
        __md_float4& operator=(__md_float4&& p)
        {
            if (this == &p) { return *this; }
            this.x = p.x; this.y = p.y;
            this.z = p.z; this.w = p.w;
            p.x = 0; p.y = 0; p.z = 0; p.w = 0;
            return *this;
        }
    };
}

