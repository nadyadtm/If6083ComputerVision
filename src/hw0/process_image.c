#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
    // TODO Fill this in
    int row = y;
    int column = x;
    int channel = c;
    if (x<0){
        column = 0;
    }
    if (y<0){
        row = 0;
    }
    if (x>im.w){
        column = im.w-1;
    }
    if (y>im.h){
        row = im.h-1;
    }
    if (c<0){
        channel = 0;
    }
    if (c>im.c){
        channel = im.c-1;
    }
    return im.data[im.w*row + column + im.w*im.h*channel];
}

void set_pixel(image im, int x, int y, int c, float v)
{
    // TODO Fill this in
        // TODO Fill this in
    int row = y;
    int column = x;
    int channel = c;
    if (x<0){
        column = 0;
    }
    if (y<0){
        row = 0;
    }
    if (x>im.w){
        column = im.w-1;
    }
    if (y>im.h){
        row = im.h-1;
    }
    if (c<0){
        channel = 0;
    }
    if (c>im.c){
        channel = im.c-1;
    }
    if (x<0 || y<0 || x>im.w || y>im.h){
        return;
    } else{
        *(im.data + (im.w*row + column + im.w*im.h*channel)) = v;
    }
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    // TODO Fill this in
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            for(int c=0;c<im.c;c++){
                float v = get_pixel(im,x,y,c);
                set_pixel(copy, x,y,c,v);
            }
        }
    }
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            float r = get_pixel(im,x,y,0)*0.299;
            float g = get_pixel(im,x,y,1)*0.587;
            float b = get_pixel(im,x,y,2)*0.114;
            float gr = r + g + b;
            set_pixel(gray,x,y,0,gr);
        }
    }
    // TODO Fill this in
    return gray;
}

void shift_image(image im, int c, float v)
{
    // TODO Fill this in
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            float shift = get_pixel(im,x,y,c)+v;
            set_pixel(im,x,y,c,shift);
        }
    }
}

void clamp_image(image im)
{
    // TODO Fill this in
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            for(int c=0;c<im.c;c++){
                float v = get_pixel(im,x,y,c);
                if (v<0){
                    set_pixel(im, x,y,c,0);
                } else if (v>1){
                    set_pixel(im, x,y,c,1);
                }
            }
        }
    }
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
    // TODO Fill this in
    float r,g,b,h,H,S,V,m,C;
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            r = get_pixel(im,x,y,0);
            g = get_pixel(im,x,y,1);
            b = get_pixel(im,x,y,2);
            V = three_way_max(r,g,b);
            m = three_way_min(r,g,b);
            C = V - m;
            if (V!=0){
                S = C / V;
            }
            else{
                S = 0;
            }

            if (C!=0){
                if (V==r){
                    h = (g-b)/C;
                }
                else if (V==g){
                    h = ((b-r)/C)+2;
                }
                else if(V==b){
                    h = ((r-g)/C)+4;
                }
                if (h<0){
                    H = (h/6) + 1;
                }
                else{
                    H = (h/6);
                }
            } else{
                H = 0;
            }
            set_pixel(im,x,y,0,H);
            set_pixel(im,x,y,1,S);
            set_pixel(im,x,y,2,V);
        }
    }
}

void hsv_to_rgb(image im)
{
    // TODO Fill this in
    float r,g,b,h,H,S,V,m,C,X;
    r=0;
    g=0;
    b=0;
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            H = get_pixel(im,x,y,0);
            S = get_pixel(im,x,y,1);
            V = get_pixel(im,x,y,2);
            C = V * S;
            h = H * 6;
            X = C * (1-fabs(fmod(h,2)-1));
            if (h>=0 && h<=1){
                r = C;
                g = X;
                b = 0;
            }
            else if (h>1 && h<=2){
                r = X;
                g = C;
                b = 0;
            }
            else if (h>2 && h<=3){
                r = 0;
                g = C;
                b = X;
            }
            else if (h>3 && h<=4){
                r = 0;
                g = X;
                b = C;
            }
            else if (h>4 && h<=5){
                r = X;
                g = 0;
                b = C;
            }
            else if (h>5 && h<=6){
                r = C;
                g = 0;
                b = X;
            }
            m = V - C;
            r = r+m;
            g = g+m;
            b = b+m;
            set_pixel(im,x,y,0,r);
            set_pixel(im,x,y,1,g);
            set_pixel(im,x,y,2,b);
        }
    }
}

void scale_image(image im, int c, float v){
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            float shift = get_pixel(im,x,y,c)*v;
            set_pixel(im,x,y,c,shift);
        }
    }
}