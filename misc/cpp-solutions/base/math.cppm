export module md.math;

namespace md::math
{
	// vector4, point4 template
	export template <typename float_t = float>
	struct float4
	{
		float_t x{0};
		float_t y{0};
		float_t z{0};
		float_t w{0};

		float4() = default;
		float4(const float4& p)
		{
			if (this == &p) { return *this; }
			x = p.x;
			y = p.y;
			z = p.z;
			w = p.w;
			return *this;
		}
		float4(float4&& p)
		{
			if (this == &p) { return *this; }
			this.x = p.x; this.y = p.y;
			this.z = p.z; this.w = p.w;
			p.x = 0; p.y = 0; p.z = 0; p.w = 0;
			return *this;
		}
		float4(float_t _x, float_t _y, float_t _z, float_t _w) : x(_x), y(_y), z(_z), w(_x) {}
		float4(float_t arr[4]) : x(arr[0]), y(arr[1]), z(arr[2]), w(arr[3]) {}

		float4 operator+(const float4& p) const
		{
			return float4(x + p.x, y + p.y, z + p.z, w + p.w);
		}
		float4& operator+=(const float4& p)
		{
			x += p.x;
			y += p.y;
			z += p.z;
			w += p.w;
			return *this;
		}
		float4 operator-(const float4& p) const
		{
			return float4(x - p.x, y - p.y, z - p.z, w - p.w);
		}
		float4& operator-=(const float4& p)
		{
			x -= p.x;
			y -= p.y;
			z -= p.z;
			w -= p.w;
			return *this;
		}
		float4& operator=(const float4& p)
		{
			if (this == &p) { return *this; }
			x = p.x;
			y = p.y;
			z = p.z;
			w = p.w;
			return *this;
		}
		float4& operator=(float4&& p)
		{
			if (this == &p) { return *this; }
			this.x = p.x; this.y = p.y;
			this.z = p.z; this.w = p.w;
			p.x = 0; p.y = 0; p.z = 0; p.w = 0;
			return *this;
		}
	};

	



}

