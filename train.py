import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator, Discriminator
from datasets.facade_dataset import Dataset

import numpy as np
import cv2

import argparse

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def tanh_range2uint_color(img):
    return (img * 128 + 128).astype(np.uint8)

def modelimg2cvimg(img):
    cvimg = np.array(img[0,:,:,:]).transpose(1,2,0)
    return tanh_range2uint_color(cvimg)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", default=200, type=int)
parser.add_argument("--iterate", default=10, type=int)
parser.add_argument("--lambda1", default=100, type=int)

args = parser.parse_args()

batchsize = 1
input_channel = 3
output_channel = 3
input_height = input_width = output_height = output_width = 256

input_data = Dataset(data_start = 1, data_end = 299)
train_len = input_data.len()

generator_G = Generator(input_channel, output_channel)
discriminator_D = Discriminator(input_channel, output_channel)
weights_init(generator_G)
weights_init(discriminator_D)

generator_G.cuda()
discriminator_D.cuda()

loss_L1 = nn.L1Loss().cuda()
loss_binaryCrossEntropy = nn.BCELoss().cuda()

optimizer_G= torch.optim.Adam(generator_G.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay= 0.00001)
optimizer_D= torch.optim.Adam(discriminator_D.parameters(), lr= 0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)
input_real_np = np.zeros((batchsize, output_channel, output_height, output_width)).astype(np.float32)

for epoch in range(args.epoch):
    for iterate in range(train_len):
    
        for i in range(batchsize):
            batch = input_data.get_image(iterate)
            input_x_np[i,:] = np.asarray(batch[0])
            input_real_np[i,:] = np.asarray(batch[1])

        input_x = Variable(torch.from_numpy(input_x_np)).cuda()
        input_real = Variable(torch.from_numpy(input_real_np)).cuda()

        out_generator_G = generator_G.forward(input_x)

        optimizer_D.zero_grad()
        negative_examples = discriminator_D.forward(input_x.detach(), out_generator_G.detach())
        positive_examples = discriminator_D.forward(input_x, input_real)
        loss_dis = 0.5 * ( loss_binaryCrossEntropy(positive_examples, Variable(torch.ones(positive_examples.size())).cuda()) \
                          +loss_binaryCrossEntropy(negative_examples, Variable(torch.zeros(negative_examples.size())).cuda()))
        loss_dis.backward(retain_variables=True)
        optimizer_D.step()

        optimizer_G.zero_grad()
        negative_examples = discriminator_D.forward(input_x, out_generator_G)
        loss_gen = loss_binaryCrossEntropy(negative_examples, Variable(torch.ones(negative_examples.size())).cuda()) \
                  +loss_L1(out_generator_G, input_real) * args.lambda1
        loss_gen.backward()
        optimizer_G.step()

        if iterate % args.iterate == 0:
            print ('{} [{}/{}] LossGen= {} LossDis= {}'.format(iterate, epoch+1, args.epoch, loss_gen.data[0], loss_dis.data[0]))

        #""" MONITOR
        out_gen = out_generator_G.cpu()
        out_gen = out_gen.data.numpy()
        cvimg = modelimg2cvimg(out_gen)
        cv2.imwrite("./results/trainGenImg%d.jpg"%iterate, cvimg)
        #"""

    torch.save(generator_G.state_dict(),'./generator_G.pth')
    torch.save(discriminator_D.state_dict(),'./discriminator_D.pth')
    input_data.shuffle()



