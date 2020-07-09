import sys

import cv2
import numpy as np


def main(arg):
    # Loads an image
    src = cv2.imread(r'hough_heap\test.jpg')
    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        return -1

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=9, param2=9,
                               minRadius=1, maxRadius=11)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 1)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 0), 2)

            # font
            font = cv2.FONT_ITALIC

            # org
            org = center

            # fontScale
            font_scale = 1

            # Blue color in BGR
            color = (255, 12, 255)

            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            src = cv2.putText(src, str(center), org, font,
                              font_scale, color, thickness, cv2.LINE_AA)

            print(radius)
            print(center)
            print()

    cv2.imshow("detected circles", src)
    cv2.waitKey(0)

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
