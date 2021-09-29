#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
    // TODO Fill in
    return get_pixel(im,round(x),round(y),c);
}

image nn_resize(image im, int w, int h)
{
    // TODO Fill in (also fix that first line)
    image resized = make_image(w, h, im.c);
    float ax = (float) im.w/w;
    float ay = (float) im.h/h;

    float bx = (ax*0.5)-0.5;
    float by = (ay*0.5)-0.5;

    for (int x=0;x<resized.w;x++){
        for (int y=0;y<resized.h;y++){
            for (int c=0;c<resized.c;c++){
                float x_mapped = ax*x+bx;
                float y_mapped = ay*y+by;
                float v = nn_interpolate(im,x_mapped,y_mapped,c);
                set_pixel(resized,x,y,c,v);
            }
        }
    }
    return resized;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
    // TODO
    float d2 = ceil(x)-x;
    float d1 = 1.0 - d2;
    float d4 = ceil(y)-y;
    float d3 = 1.0 - d4;
    
    float q1,q2,q,v1,v2,v3,v4;
    v1 = get_pixel(im,floor(x),floor(y),c);
    v2 = get_pixel(im,ceil(x),floor(y),c);
    v3 = get_pixel(im,floor(x),ceil(y),c);
    v4 = get_pixel(im,ceil(x),ceil(y),c);
    q1 = (v1*d2) + (v2*d1);
    q2 = (v3*d2) + (v4*d1);
    q = (q1*d4) + (q2*d3);
    return q;
}

image bilinear_resize(image im, int w, int h)
{
    // TODO
    image resized = make_image(w, h, im.c);
    float ax = (float) im.w/w;
    float ay = (float) im.h/h;

    float bx = (ax*0.5)-0.5;
    float by = (ay*0.5)-0.5;

    for (int x=0;x<resized.w;x++){
        for (int y=0;y<resized.h;y++){
            for (int c=0;c<resized.c;c++){
                float x_mapped = ax*x+bx;
                float y_mapped = ay*y+by;
                float v = bilinear_interpolate(im,x_mapped,y_mapped,c);
                set_pixel(resized,x,y,c,v);
            }
        }
    }
    return resized;
}

