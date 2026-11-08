#include <cstdio>
#include <mdmath/mdmath.h>

using vec3 = md::math::__md_vec<float32>;

int main()
{
	vec3 v1(1, 0.5, 0);
	vec3 v2(0, 1, 0);

	vec3 c = v1.cross(v2);
	vec3 c1 = v2.cross(v1);
	
	printf("%f\n", vec3::dot(v1, v2));
	printf("%f, %f, %f\n", c.x, c.y, c.z);
	printf("%f, %f, %f\n", c1.x, c1.y, c1.z);
}