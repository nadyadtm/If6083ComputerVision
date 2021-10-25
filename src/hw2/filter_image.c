#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
    // TODO
    for (int c=0;c<im.c;c++){
        float sum = 0;
        for (int x=0;x<im.w;x++){
            for (int y=0;y<im.h;y++){
                sum = sum+get_pixel(im,x,y,c);
            }
        }
        for (int x=0;x<im.w;x++){
            for (int y=0;y<im.h;y++){
                float v = get_pixel(im,x,y,c);
                if (sum!=0){
                    set_pixel(im,x,y,c,v/sum);
                }
                else{
                    set_pixel(im,x,y,c,1.0/im.w*im.h);
                }
            }
        }
    }
}

image make_box_filter(int w)
{
    // TODO
    image im = make_image(w,w,1);
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            set_pixel(im,x,y,0,1.0);
        }
    }
    l1_normalize(im);
    return im;
}

image convolve_image(image im, image filter, int preserve)
{
    // TODO
	assert(filter.c==1 || im.c ==filter.c);

    int coor_pad_x = (filter.w-1)/2;
    int coor_pad_y = (filter.h-1)/2;
    image convolved;

    if (preserve==1){
        convolved = make_image(im.w,im.h,im.c);
        for (int c=0;c<im.c;c++){

            for (int x=0;x<im.w;x++){
                for (int y=0;y<im.h;y++){

                    float sum_filter = 0;
                    for (int i=0; i<filter.w;i++){
                        for (int j=0; j<filter.h;j++){
                            int filter_c;
                            if (filter.c==1){
                                filter_c = 0;
                            }
                            else{
                                filter_c = c;
                            }
                            sum_filter = sum_filter + get_pixel(filter,i,j,filter_c)*get_pixel(im,i-coor_pad_x+x,j-coor_pad_y+y,c);
                        }
                    }

                    set_pixel(convolved,x,y,c,sum_filter);
                }  
            }
        }
    } 
    else {
        convolved = make_image(im.w,im.h,1);
        for (int x=0;x<im.w;x++){
            for (int y=0;y<im.h;y++){

                float sum_filter = 0;
                for (int c=0;c<im.c;c++){
                    for (int i=0; i<filter.w;i++){
                        for (int j=0; j<filter.h;j++){
                            int filter_c;
                            if (filter.c==1){
                                filter_c = 0;
                            }
                            else{
                                filter_c = c;
                            }
                            sum_filter = sum_filter + get_pixel(filter,i,j,c)*get_pixel(im,i-coor_pad_x+x,j-coor_pad_y+y,c);
                        }
                    }
                }

                set_pixel(convolved,x,y,0,sum_filter);
            }  
        }
    }

    return convolved;
}

image make_highpass_filter()
{
    // TODO
    image filter = make_image(3,3,1);
    set_pixel(filter, 0,0,1,0);
    set_pixel(filter, 1,0,1,-1);
    set_pixel(filter, 2,0,1,0);
    set_pixel(filter, 0,1,1,-1);
    set_pixel(filter, 1,1,1,4);
    set_pixel(filter, 2,1,1,-1);
    set_pixel(filter, 0,2,1,0);
    set_pixel(filter, 1,2,1,-1);
    set_pixel(filter, 2,2,1,0);
    return filter;
}

image make_sharpen_filter()
{
    // TODO
    image filter = make_image(3,3,1);
    set_pixel(filter, 0,0,1,0);
    set_pixel(filter, 1,0,1,-1);
    set_pixel(filter, 2,0,1,0);
    set_pixel(filter, 0,1,1,-1);
    set_pixel(filter, 1,1,1,5);
    set_pixel(filter, 2,1,1,-1);
    set_pixel(filter, 0,2,1,0);
    set_pixel(filter, 1,2,1,-1);
    set_pixel(filter, 2,2,1,0);
    return filter;
}

image make_emboss_filter()
{
    // TODO
    image filter = make_image(3,3,1);
    set_pixel(filter, 0,0,1,-2);
    set_pixel(filter, 1,0,1,-1);
    set_pixel(filter, 2,0,1,0);
    set_pixel(filter, 0,1,1,-1);
    set_pixel(filter, 1,1,1,1);
    set_pixel(filter, 2,1,1,1);
    set_pixel(filter, 0,2,1,0);
    set_pixel(filter, 1,2,1,1);
    set_pixel(filter, 2,2,1,2);
    return filter;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: TODO
/* Filter yang tidak perlu menggunakan preserve adalah highpass, karena jika di preserve maka edgenya tidak terlihat
secara signifikan sehingga tidak perlu menggunakan preserve. Sedangkan sharpen dan emboss perlu digunakan perserve karena tujuan dari
sharpen adalah menajamkan gambar dan tujuan dari emboss adalah memberikan efek timbul pada gambar sehingga diperlukan hasil konvolusi
yang sama dengan gambar aslinya*/

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: TODO
/* Perlu dilakukan clamp image untuk menormalisasikan imagenya, karena pada saat melakukan konvolusi ada kemungkinan nilainya
melebihi batas sehingga perlu dilakukan clamp image agar nilainya sesuai dengan rangenya*/

image make_gaussian_filter(float sigma)
{
    // TODO
    image filter;
    int size = (int)roundf(sigma * 6);
    if (size % 2 == 0)
        size=size+1;
    filter = make_image(size,size,1);
    int center = filter.w/2;
    for (int x=0;x<filter.w;x++){
        for (int y=0;y<filter.h;y++){
            float xpow = (x-center)*(x-center);
            float ypow = (y-center)*(y-center);
            float plus = -(xpow + ypow);
            float twosigma = sigma*sigma;
            float exponen = exp(plus/(2*twosigma));
            float per2pi = 1/(twosigma*TWOPI);
            float gxy = per2pi * exponen;
            set_pixel(filter,x,y,0,gxy);
        }
    }
    l1_normalize(filter);
    return filter;
}

image add_image(image a, image b)
{
    // TODO
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image new_image = make_image(a.w,a.h,a.c);
    for (int c=0;c<a.c;c++){

        for (int x=0;x<a.w;x++){
            for (int y=0;y<a.h;y++){
                float pA = get_pixel(a,x,y,c);
                float pB = get_pixel(b,x,y,c);
                set_pixel(new_image,x,y,c,pA+pB);
            }  
        }
    }
    return new_image;
}

image sub_image(image a, image b)
{
    // TODO
    assert(a.w == b.w && a.h == b.h && a.c == b.c);
    image new_image = make_image(a.w,a.h,a.c);
    for (int c=0;c<a.c;c++){

        for (int x=0;x<a.w;x++){
            for (int y=0;y<a.h;y++){
                float pA = get_pixel(a,x,y,c);
                float pB = get_pixel(b,x,y,c);
                set_pixel(new_image,x,y,c,pA-pB);
            }  
        }
    }
    return new_image;
}

image make_gx_filter()
{
    // TODO
    image filter = make_image(3,3,1);
    set_pixel(filter, 0,0,1,-1);
    set_pixel(filter, 1,0,1,0);
    set_pixel(filter, 2,0,1,1);
    set_pixel(filter, 0,1,1,-2);
    set_pixel(filter, 1,1,1,0);
    set_pixel(filter, 2,1,1,2);
    set_pixel(filter, 0,2,1,-1);
    set_pixel(filter, 1,2,1,0);
    set_pixel(filter, 2,2,1,1);
    return filter;
}

image make_gy_filter()
{
    // TODO
    image filter = make_image(3,3,1);
    set_pixel(filter, 0,0,1,-1);
    set_pixel(filter, 1,0,1,-2);
    set_pixel(filter, 2,0,1,-1);
    set_pixel(filter, 0,1,1,0);
    set_pixel(filter, 1,1,1,0);
    set_pixel(filter, 2,1,1,0);
    set_pixel(filter, 0,2,1,1);
    set_pixel(filter, 1,2,1,2);
    set_pixel(filter, 2,2,1,1);
    return filter;
}

void feature_normalize(image im)
{
    // TODO
    float min = get_pixel(im,0,0,0);
    float max = min;
    for (int c=0;c<im.c;c++){
        for (int x=0;x<im.w;x++){
            for (int y=0;y<im.h;y++){
                float v = get_pixel(im,x,y,c);
                if (v>max){
                    max = v;
                }
                else if (v<min){
                    min = v;
                }
            }  
        }
    }

    float range = max-min;

    if (range>0){
        for (int c=0;c<im.c;c++){
            for (int x=0;x<im.w;x++){
                for (int y=0;y<im.h;y++){
                    float v = get_pixel(im,x,y,c);
                    float normal = (v-min)/range;
                    set_pixel(im,x,y,c,normal);
                }  
            }
            
        }
    }
    else{
        for (int c=0;c<im.c;c++){
            for (int x=0;x<im.w;x++){
                for (int y=0;y<im.h;y++){
                    set_pixel(im,x,y,c,0);
                }  
            }
            
        }
    }
}

image *sobel_image(image im)
{
    // TODO
    image magnitude = make_image(im.w, im.h, 1);
    image orientation = make_image(im.w, im.h, 1);
    image gx = convolve_image(im,make_gx_filter(),0);
    image gy = convolve_image(im,make_gy_filter(),0);

    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){

            float gx_el = get_pixel(gx,x,y,0);
            float gy_el = get_pixel(gy,x,y,0);
            set_pixel(magnitude,x,y,0,sqrt((gx_el*gx_el)+(gy_el*gy_el)));
            set_pixel(orientation,x,y,0,atan2(gy_el,gx_el));

        }  
    }

    image* result = (image*) malloc(sizeof(image) * 2);
    result[0]=magnitude;
    result[1]=orientation;

    return result;
}

image colorize_sobel(image im)
{
    // TODO
    image* ret = sobel_image(im);
    image magnitude = ret[0];
    image orientation = ret[1];
    feature_normalize(magnitude);
    feature_normalize(orientation);
    image new_image = make_image(im.w,im.h,im.c);

    float r,g,b,h,H,S,V,m,C,X;
    r=0;
    g=0;
    b=0;
    for (int x=0;x<im.w;x++){
        for (int y=0;y<im.h;y++){
            H = get_pixel(orientation,x,y,0);
            S = get_pixel(magnitude,x,y,0);
            V = get_pixel(magnitude,x,y,0);
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
            set_pixel(new_image,x,y,0,r);
            set_pixel(new_image,x,y,1,g);
            set_pixel(new_image,x,y,2,b);
        }
    }
    return new_image;
}
