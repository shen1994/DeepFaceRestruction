#include "mesh_core.h"


/* Judge whether the point is in the triangle
Method:
    http://blackpawn.com/texts/pointinpoly/
Args:
    point: [x, y] 
    tri_points: three vertices(2d points) of a triangle. 2 coords x 3 vertices
Returns:
    bool: true for in triangle
*/
bool isPointInTri(point p, point p0, point p1, point p2)
{   
    // vectors
    point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if(dot00*dot11 - dot01*dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1/(dot00*dot11 - dot01*dot01);

    float u = (dot11*dot02 - dot01*dot12)*inverDeno;
    float v = (dot00*dot12 - dot01*dot02)*inverDeno;

    // check if point in triangle
    return (u >= 0) && (v >= 0) && (u + v < 1);
}

void _render_texture_core(
    float* image, float* vertices, int* triangles, 
	float* tri_depth, float* tri_tex, float* depth_buffer,
	int ver_len, int tri_len, int tex_len, int h, int w, int c)
{
    int i;
    int x, y, k;
	int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    point p, p0, p1, p2;
	float p0_depth, p1_depth, p2_depth;
    int x_min, x_max, y_min, y_max;

    for(i = 0; i < tri_len; i++)
    {
        tri_p0_ind = triangles[i];
        tri_p1_ind = triangles[tri_len + i];
        tri_p2_ind = triangles[tri_len * 2 + i];

		p0.x = vertices[tri_p0_ind]; p0.y = vertices[ver_len + tri_p0_ind]; p0_depth = vertices[ver_len * 2 + tri_p0_ind];
		p1.x = vertices[tri_p1_ind]; p1.y = vertices[ver_len + tri_p1_ind]; p1_depth = vertices[ver_len * 2 + tri_p1_ind];
		p2.x = vertices[tri_p2_ind]; p2.y = vertices[ver_len + tri_p2_ind]; p2_depth = vertices[ver_len * 2 + tri_p2_ind];
		
		x_min = max((int)ceil(min(p0.x, min(p1.x, p2.x))), 0);
		x_max = min((int)floor(max(p0.x, max(p1.x, p2.x))), w - 1);
      
		y_min = max((int)ceil(min(p0.y, min(p1.y, p2.y))), 0);
		y_max = min((int)floor(max(p0.y, max(p1.y, p2.y))), h - 1);

        if(x_max < x_min || y_max < y_min)
        {
            continue;
        }

        for(y = y_min; y <= y_max; y++) //h
        {
            for(x = x_min; x <= x_max; x++) //w
            {
				p.x = x; p.y = y;
                if(tri_depth[i] > depth_buffer[y*w+x] && isPointInTri(p, p0, p1, p2))
                {   
					for(k = 0; k < c; k++)
                    { 
						image[y*w*c + x*c + k] = tri_tex[k*tex_len+i];
                    }
					depth_buffer[y*w + x] = tri_depth[i];
                }
            }
        }
    }
}
