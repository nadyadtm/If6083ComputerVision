from uwimg import *
im = load_image("data/dog.jpg")

# -----------2.2 Write a convolution function---------------
# Perform convolution
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
save_image(blur, "dog-box7")

# Resize the picture
f = make_box_filter(7)
blur = convolve_image(im, f, 1)
thumb = nn_resize(blur, blur.w//7, blur.h//7)
save_image(thumb, "dogthumb")

# Try Other Filter (To answer question 2.2.1 and 2.2.2)
# Sharpen Filter
f = make_sharpen_filter()
blur = convolve_image(im, f, 1)
clamp_image(blur)
save_image(blur, "dog-sharpen")

# Emboss Filter
f = make_emboss_filter()
blur = convolve_image(im, f, 1)
clamp_image(blur)
save_image(blur, "dog-emboss")

# HighPass Filter (Preserved)
f = make_highpass_filter()
blur = convolve_image(im, f, 1)
clamp_image(blur)
save_image(blur, "dog-highpass-preserved")

# HighPass Filter
f = make_highpass_filter()
blur = convolve_image(im, f, 0)
clamp_image(blur)
save_image(blur, "dog-highpass")

# ---------------2.3 Implement a Gaussian Kernel---------------
f = make_gaussian_filter(2)
blur = convolve_image(im, f, 1)
save_image(blur, "dog-gauss2")

# ---------------------2.4 Hybrid Image------------------------
f = make_gaussian_filter(2)
lfreq = convolve_image(im, f, 1)
hfreq = im - lfreq
reconstruct = lfreq + hfreq
save_image(lfreq, "low-frequency")
save_image(hfreq, "high-frequency")
save_image(reconstruct, "reconstruct")

# ----------------Ron and Dumbledore Fusion--------------------
im1 = load_image("data/ron.png")
im2 = load_image("data/dumbledore.png")
f1 = make_gaussian_filter(3)
f2 = make_gaussian_filter(2)
lfreq = convolve_image(im2, f1, 1)
lfreqron = convolve_image(im1, f2, 1)
hfreq = im1 - lfreqron
reconstruct = lfreq + hfreq
clamp_image(reconstruct)
save_image(reconstruct, "ronbledore")

# -------2.5.3 Calculate Gradient Magnitude and Direction-------
res = sobel_image(im)
mag = res[0]
ori = res[1]
feature_normalize(mag)
feature_normalize(ori)
save_image(mag, "magnitude")
save_image(ori, "orientation")

# -----------2.5.4 Make a Colorized Representation--------------
color_sobel = colorize_sobel(im)
color_sobel = convolve_image(color_sobel,make_gaussian_filter(1),1)
save_image(color_sobel, "colorize_sobel")
