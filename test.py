import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator
from datasets.facade_dataset import Dataset

import numpy as np
import cv2

batchsize =1
input_channel = 3
output_channel= 3
input_height = input_width = 256

input_data = Dataset(data_start = 300, data_end = 378)
test_len = input_data.len()

generator_G = Generator(input_channel, output_channel).cuda()
generator_G.load_state_dict(torch.load('./generator_G.pth'))

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)

def tanh_range2uint_color(img):
    return (img * 128 + 128).astype(np.uint8)

def modelimg2cvimg(img):
    cvimg = np.array(img[0,:,:,:]).transpose(1,2,0)
    return tanh_range2uint_color(cvimg)

for iterate in range(test_len):
    image = input_data.get_image(iterate)
    input_x_np[0,:] = np.asarray(image[0])

    input_x = Variable(torch.from_numpy(input_x_np)).cuda()
    out_generator_G = generator_G.forward(input_x)

    out_gen = out_generator_G.cpu()
    out_gen = out_gen.data.numpy()
    cvimg = modelimg2cvimg(out_gen)
    cv2.imwrite("./results/testGenImg%d.jpg"%iterate, cvimg)

    cvimg = modelimg2cvimg(input_x_np)
    cv2.imwrite("./results/testInputImg%d.jpg"%iterate, cvimg)

