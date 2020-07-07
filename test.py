# Test
import cv2

from image_processing import ImageProcessing, bilateral_filter
from image_processing import convert_color, midpoint, blur, median_blur
from image_processing import info, show, contrast, threshold, draw_contour

image = ImageProcessing(r'images_heap\sym.png')
image.read()

info(image.img)
show(image.img)

# contrast
show(contrast(image.img, 3, 100))

# convert color
img = convert_color(image.img)

# threshold image
thresh = threshold(img, 55, cv2.THRESH_BINARY_INV)
show(thresh)

# midpoint on the original image
show(midpoint(img, 5))

# midpoint on the threshold image
show(midpoint(thresh, 3))

# blur on the threshold image
show(blur(thresh, 5))

# median blur on the original image
show(median_blur(img, 5))

# draw contours
test_image = img
draw_contour(median_blur(img, 5), test_image)
show(test_image)
# bilateral filter on the original image
show(bilateral_filter(thresh, 160, 9))
