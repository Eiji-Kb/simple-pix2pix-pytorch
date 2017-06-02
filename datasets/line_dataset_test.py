import numpy as np
import cv2
import random

class Dataset():
    def __init__(self, dataDir='./datasets/cnvframesEx', data_start = 1, data_end = 1000):

        self.dataset = []

        for i in range (data_start, data_end + 1):
            input_x_image = cv2.imread(dataDir+"/%08dA.jpg"%i)

            self.dataset.append(input_x_image)

    def uint_color2tanh_range(self, img):
        return img / 128. - 1.

    def len(self):
        return len(self.dataset)

    def get_image(self, i):
        input_x_image = np.asarray(self.dataset[i])
        input_x_image = self.uint_color2tanh_range(input_x_image.astype(np.float32))

        return input_x_image.transpose(2,0,1)

