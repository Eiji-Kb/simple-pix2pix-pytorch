#import os
import numpy as np
import cv2
import random
#import matplotlib.pyplot as plt


class LineDataset():
    def __init__(self, dataDir='./cnvframesEx', data_start = 1, data_end = 1000, height = 256, width = 256):

        self.dataset = []
        self.shufflelist = []

        for i in range (data_start, data_end):
            real_image = cv2.imread(dataDir+"/%08dA.jpg"%i)
            real_image = self.resize(real_image)

            input_x_image = cv2.imread(dataDir+"/%08dB.jpg"%i)
            input_x_image = self.resize(input_x_image)

            self.dataset.append((input_x_image, real_image))
            
            self.shufflelist = list(range(self.len()))
      
    def resize (self, img, base_size = 286, ipl_alg=cv2.INTER_CUBIC):
        img_height, img_width, _ = img.shape
        short_side = min(img_height, img_width)
        rasio = base_size / short_side

        new_width  = int(img_width * rasio)
        new_height = int(img_height * rasio)

        return cv2.resize(img, (new_height, new_width), interpolation=ipl_alg)

    def uint_color2tanh_range(self, img):
        return img / 128. - 1.
      
    def len(self):
        return len(self.dataset)

    def shuffle(self):
        random.shuffle(self.shufflelist)
    
    def get_image(self, i):
        input_x_image = np.asarray(self.dataset[self.shufflelist[i]][0])
        real_image = np.asarray(self.dataset[self.shufflelist[i]][1])

        input_x_image, real_image = self.crip_imgs_pair(input_x_image, real_image)
        
        if random.random() > 0.5:
            input_x_image = cv2.flip(input_x_image, 1)
            real_image = cv2.flip(real_image, 1)        

        input_x_image = self.uint_color2tanh_range(input_x_image.astype(np.float32))
        real_image = self.uint_color2tanh_range(real_image.astype(np.float32))

        return input_x_image.transpose(2,0,1), real_image.transpose(2,0,1)

    def crip_imgs_pair(self, img1, img2, crip_size = 256):
        img_height, img_width, _ = img1.shape
        crip_x = random.randint(0, img_width - crip_size)
        crip_y = random.randint(0, img_height - crip_size)

        return img1[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :], img2[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :]

    def crip_img(self, img, crip_size = 256): 
        img_height, img_width, _ = img.shape
        crip_x = random.randint(0, img_width - crip_size)
        crip_y = random.randint(0, img_height - crip_size)

        return img[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :]

