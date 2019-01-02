#ifndef MESH_CORE_HPP_
#define MESH_CORE_HPP_

#include <stdio.h>
#include <cmath>
#include <algorithm>  
#include <string>
#include <iostream>
#include <fstream>

using namespace std;

class point
{
 public:
    float x;
    float y;

    float dot(point p)
    {
        return this->x * p.x + this->y * p.y;
    }

    point operator-(const point& p)
    {
        point np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    }

    point operator+(const point& p)
    {
        point np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    }

    point operator*(float s)
    {
        point np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    }
}; 


bool isPointInTri(point p, point p0, point p1, point p2, int h, int w);

void _render_texture_core(
    float* image, float* vertices, int* triangles, 
	float* tri_depth, float* tri_tex, float* depth_buffer,
    int ver_len, int tri_len, int tex_len, int h, int w, int c);

#endif