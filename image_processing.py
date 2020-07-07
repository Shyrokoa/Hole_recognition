# import modules
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, minimum_filter


class ImageProcessing:

    def __init__(self, path_to_file):
        self.path = path_to_file
        self.img = ''
        self.read()

    def read(self):
        self.img = cv2.imread(self.path, 1)


def info(array):
    print(f'Image shape: {array.shape}, dtype: {array.dtype}')


def show(array):
    plt.figure(figsize=(20, 20))
    plt.imshow(array, cmap="gray")
    plt.show()


# remove the background noise
def threshold(array, target, method):
    _, thresh = cv2.threshold(array, target, 255, method)
    return thresh


# filter n by n
def midpoint(array, n):
    maxf = maximum_filter(array, (n, n))
    minf = minimum_filter(array, (n, n))
    mid_point = (maxf + minf) / 2
    return mid_point


# blur
def blur(array, n):
    return cv2.blur(array, (n, n))


# median blur
def median_blur(array, n):
    return cv2.medianBlur(array, n)


# bilateral filter
# Sigma values: For simplicity, you can set the 2 sigma values to be the same.
#    If they are small (< 10), the filter will not have much effect, whereas
#    if they are large (> 150), they will have a very strong effect, making the
#    image look “cartooning”.
# Filter size: Large filters (d > 5) are very slow, so it is recommended to use
#    d=5 for real-time applications, and perhaps d=9 for offline applications
#    that need heavy noise filtering.
def bilateral_filter(array, sigma, n):
    return cv2.bilateralFilter(array, sigma, n, n)


def draw_contour(array, destination):
    # self.contours = []
    contours, hierarchy = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in contours:
        cv2.drawContours(destination, [cnt], 0, (255, 0, 0), 2)
        # self.contours.append(cnt)
        # print(cnt)


def contrast(array, alpha, beta):
    # alpha - Contrast control (1.0-3.0)
    # beta  - Brightness control (0-100)
    return cv2.convertScaleAbs(array, alpha=alpha, beta=beta)


def convert_color(array):
    return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
