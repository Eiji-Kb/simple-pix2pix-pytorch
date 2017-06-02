import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator
from datasets.line_dataset_test import Dataset

import numpy as np
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_start", default=1000, type=int)
parser.add_argument("--data_end", default=1500, type=int)
parser.add_argument("--dataset", default="./datasets/cnvframesEx", type=str)
args = parser.parse_args()

batchsize = 1
input_channel = 3
output_channel= 3
input_height = input_width = 256

input_data = Dataset(dataDir = args.dataset, data_start = args.data_start, data_end = args.data_end)
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
    input_x_np[0,:] = input_data.get_image(iterate)

    input_x = Variable(torch.from_numpy(input_x_np)).cuda()
    out_generator_G = generator_G.forward(input_x)

    out_gen = out_generator_G.cpu()
    out_gen = out_gen.data.numpy()
    cvimg = modelimg2cvimg(out_gen)

    cv2.imwrite("./results/testGen%d.jpg"%(iterate + args.data_start), cvimg)

