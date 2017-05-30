import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator
from linedataset import LineDataset

import numpy as np
import cv2

batchsize =1
input_channel = 3
output_channel= 3
input_height = input_width = output_height = output_width = 256

input_data = LineDataset(data_start = 1000, data_end = 1500, height = input_height, width = input_width)
test_len = input_data.len()

generator_G = Generator(input_channel, output_channel)
generator_G.load_state_dict(torch.load('./generator_G.pth'))

generator_G.cuda()

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)

for iterate in range(test_len):
    image = input_data.get_image(iterate)
    input_x_np[0,:] = np.asarray(image[0])

    input_x = Variable(torch.from_numpy(input_x_np)).cuda()

    out_generator_G = generator_G.forward(input_x)

    genout = out_generator_G.cpu()
    genout = genout.data.numpy()
    genimg = np.array(genout[0,:,:,:]).transpose(1,2,0)
    genimg = (genimg * 128 + 128).astype(np.uint8)
    cv2.imwrite("testImg%d.jpg"%iterate, genimg)

