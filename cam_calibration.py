import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Calibration:

    def __init__(self):
        self.images = ''
        self.obj_p = ''
        self.criteria = ''
        self.gray = ''
        self.cam_parameters = ''
        self.img = ''
        self.new_camera_mtx = ''
        self.roi = ''
        self.dst = ''
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.self.img_points = ''

    def setup_parameters(self):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.obj_p = np.zeros((6 * 7, 3), np.float32)
        self.obj_p[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        self.images = glob.glob('cameras_heap\dataset3\*.jpg')

    def read_image(self):
        for f_name in self.images:
            img = cv2.imread(f_name)
            self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(self.gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                self.obj_points.append(self.obj_p)
                corners2 = cv2.cornerSubPix(self.gray, corners, (11, 11), (-1, -1), self.criteria)
                self.img_points.append(corners2)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

    def get_matrix(self):
        # cam_parameters: ret, mtx, dist, rvecs, tvecs
        self.cam_parameters = cv2.calibrateCamera(self.obj_points, self.img_points, self.gray.shape[::-1], None, None)
        print(f'Mtx:\n {self.cam_parameters[1]}\nDist:\n {self.cam_parameters[2]}')

    def matrix_info(self):
        print(f'Mtx:\n {self.cam_parameters[1]}\nDist:\n {self.cam_parameters[2]}')

    def read_new_img(self, path):
        self.img = cv2.imread(path)
        h, w = self.img.shape[:2]
        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(self.cam_parameters[1],
                                                                      self.cam_parameters[2],
                                                                      (w, h),
                                                                      1,
                                                                      (w, h))

    def undistort(self):
        self.dst = cv2.undistort(self.img, self.cam_parameters[1], self.cam_parameters[2], None, self.new_camera_mtx)

        # TODO crop the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        plt.figure(figsize=(20, 20))
        plt.imshow(self.dst, cmap="gray")
        plt.show()

    def get_undistort(self):
        return self.dst


cam = Calibration()
cam.setup_parameters()
cam.read_image()
cam.get_matrix()
cam.matrix_info()
cam.read_new_img('cameras_heap\dataset3\photo_2020-06-24_13-07-34.jpg')
cam.undistort()
