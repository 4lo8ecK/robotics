#pragma once

#include <concepts>
#include <cmath>

#include "./typedefs.h"
#include "./__md_float.h"


namespace md::math
{
    template <typename float_t = float32>
    struct __md_vec : __md_float4<float_t>
    {
        using __md_float4<float_t>::__md_float4;
        
        float_t get_length() const
        {
            return std::sqrt(this->x*this->x + this->y*this->y + this->z*this->z);
        }
        float dot(const __md_vec& vec) const
        {
            return dot(*this, vec);
        }
        __md_vec cross(const __md_vec& vec)
        {
            return cross(*this, vec);
        }
        static float_t dot(const  __md_vec<float_t>& a, const  __md_vec<float_t>& b)
        {
            return a.x*b.x + a.y*b.y + a.z*b.z;
        }
        static  __md_vec<float_t> cross(const __md_vec<float_t>& a, const __md_vec<float_t>& b)
        {
            float_t x = a.y*b.z - a.z*b.y;
            float_t y = a.z*b.x - a.x*b.z;
            float_t z = a.x*b.y - a.y*b.x;
            return  __md_vec<float_t>(x, y, z);
        }
    };
}