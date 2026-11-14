#pragma once

#include "./typedefs.h"
#include <cmath>
#include <stdexcept>

struct vec2
{
	union
	{
		float_t x{0};
		float_t width;
	};
	union
	{
		float_t y{0};
		float_t height;
	};

	vec2() = default;
	vec2(float_t _x, float_t _y) : x(_x), y(_y) {}
	vec2(const vec2& p_vec): x(p_vec.x), y(p_vec.y) {}
	vec2(vec2&& p_vec)
	{
		x = p_vec.x;
		y = p_vec.y;
		p_vec.x = 0;
		p_vec.y = 0;
	}

	constexpr float_t& operator[](uint32 p_axes)
	{
		if (p_axes >= 0 && p_axes < 2) return (&x)[p_axes];
		throw std::out_of_range("Vallue must be less then 2");
	}
	constexpr const float_t& operator[](uint32 p_axes) const
	{
		if (p_axes >= 0 && p_axes < 2) return (&x)[p_axes];
		throw std::out_of_range("Vallue must be less then 2");
	}
	
	void normalize();
	vec2 normalized() const;
	bool is_normalized() const;

	float_t length() const;
	float_t length_squared() const;

	void zero() { x = y = 0; }

	float_t dot(const vec2& p_vec) const;
	float_t cross(const vec2& p_vec) const;
	vec2 project(const vec2& p_vec) const;

	float_t distance_to(const vec2& p_vec) const { (p_vec - *this).length(); }
	float_t distance_to_squared(const vec2& p_vec) const;
	float_t angle_to(const vec2& p_vec) const;

	constexpr vec2 operator+(const vec2& p_vec) const;
	constexpr void operator+=(const vec2& p_vec);
	constexpr vec2 operator-(const vec2& p_vec) const;
	constexpr void operator-=(const vec2& p_vec);

	constexpr vec2 operator*(const vec2&) const;
	constexpr vec2 operator*(float_t val) const;
	constexpr void operator*=(float_t val);
	constexpr void operator*=(const vec2& p_vec) { *this = *this * p_vec;}
	
	constexpr vec2 operator/(const vec2& p_vec) const;
	constexpr vec2 operator/(float_t val) const;
	constexpr void operator/=(const vec2& p_vec);
	constexpr void operator/=(float_t val);

	constexpr vec2 operator-() const;

	constexpr bool operator==(const vec2& p_vec) const;
	constexpr bool operator!=(const vec2& p_vec) const;

	constexpr bool operator<(const vec2& p_vec) const { return x == p_vec.x ? (y < p_vec.y): x < p_vec.x; }
	constexpr bool operator>(const vec2& p_vec) const { return x == p_vec.x ? (y > p_vec.y): x > p_vec.x; }
	constexpr bool operator<=(const vec2& p_vec) const { return x == p_vec.x ? (y <= p_vec.y): x < p_vec.x; }
	constexpr bool operator>=(const vec2& p_vec) const { return x == p_vec.x ? (y >= p_vec.y): x > p_vec.x; }

	float_t angle() const;
	static vec2 from_angle(float_t angle);

	inline vec2 abs() const { return vec2(std::abs(x), std::abs(y)); }
	vec2 clamp(const vec2& p_min, const vec2& p_max) const;

};

struct vec3
{
	float_t x{0};
	float_t y{0};
	float_t z{0};	

	vec3() = default;
	vec3(float_t _x, float_t _y) : x(_x), y(_y) {}
	vec3(const vec3& p_vec): x(p_vec.x), y(p_vec.y) {}
	vec3(vec3&& p_vec)
	{
		x = p_vec.x;
		y = p_vec.y;
		p_vec.x = 0;
		p_vec.y = 0;
	}

	constexpr float_t& operator[](uint32 p_axes)
	{
		if (p_axes >= 0 && p_axes < 3) return (&x)[p_axes];
		throw std::out_of_range("Vallue must be less then 2");
	}
	constexpr const float_t& operator[](uint32 p_axes) const
	{
		if (p_axes >= 0 && p_axes < 3) return (&x)[p_axes];
		throw std::out_of_range("Vallue must be less then 2");
	}
	
	void normalize();
	vec3 normalized() const;
	bool is_normalized() const;

	float_t length() const;
	float_t length_squared() const;

	void zero() { x = y = 0; }

	float_t dot(const vec3& p_vec) const;
	float_t cross(const vec3& p_vec) const;
	vec3 project(const vec3& p_vec) const;

	float_t distance_to(const vec3& p_vec) const { (p_vec - *this).length(); }
	float_t distance_to_squared(const vec3& p_vec) const;
	float_t angle_to(const vec3& p_vec) const;

	constexpr vec3 operator+(const vec3& p_vec) const;
	constexpr void operator+=(const vec3& p_vec);
	constexpr vec3 operator-(const vec3& p_vec) const;
	constexpr void operator-=(const vec3& p_vec);

	constexpr vec3 operator*(const vec3&) const;
	constexpr vec3 operator*(float_t val) const;
	constexpr void operator*=(float_t val);
	constexpr void operator*=(const vec3& p_vec) { *this = *this * p_vec;}
	
	constexpr vec3 operator/(const vec3& p_vec) const;
	constexpr vec3 operator/(float_t val) const;
	constexpr void operator/=(const vec3& p_vec);
	constexpr void operator/=(float_t val);

	constexpr vec3 operator-() const;

	constexpr bool operator==(const vec3& p_vec) const;
	constexpr bool operator!=(const vec3& p_vec) const;

	constexpr bool operator<(const vec3& p_vec) const { return x == p_vec.x ? (y < p_vec.y): x < p_vec.x; }
	constexpr bool operator>(const vec3& p_vec) const { return x == p_vec.x ? (y > p_vec.y): x > p_vec.x; }
	constexpr bool operator<=(const vec3& p_vec) const { return x == p_vec.x ? (y <= p_vec.y): x < p_vec.x; }
	constexpr bool operator>=(const vec3& p_vec) const { return x == p_vec.x ? (y >= p_vec.y): x > p_vec.x; }

	float_t angle() const;
	static vec3 from_angle(float_t angle);

	inline vec3 abs() const { return vec3(std::abs(x), std::abs(y)); }
	vec3 clamp(const vec3& p_min, const vec3& p_max) const;

};