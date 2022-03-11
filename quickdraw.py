import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import transform

import torch
import torch.nn as nn
from torch.autograd import Variable

from Classification.models import resnet34
from Generation.models import CDCGAN_Generator
from DataUtils.config import *

if __name__ == "__main__":
    if not os.path.isdir("./trained_models/"):
        os.mkdir("./trained_models/")
    if not os.path.isdir("./trained_models/generator/"):
        os.mkdir("./trained_models/generator/")
    if not os.path.isdir("./trained_models/generator/csv"):
        os.mkdir("./trained_models/generator/csv")
    if not os.path.isdir("./trained_models/generator/image"):
        os.mkdir("./trained_models/generator/image/")

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--output_num", type = int, default = 10, help = "number of output image")
    args = vars(ap.parse_args())

    generator = CDCGAN_Generator(NUM_CLASS)
    generator.load_state_dict(torch.load("./trained_models/generator/generator.pth"))
    generator.cuda()
    print("Generator model loaded.")

    print("Images generation starting.")
    # fixed noise & label
    fixed_z_noise = torch.randn(args["output_num"], NUM_CLASS * NUM_CLASS)
    fixed_y = torch.zeros(args["output_num"], 1)
    for i in range(NUM_CLASS - 1):
        temp_z_noise = torch.randn(args["output_num"], NUM_CLASS * NUM_CLASS)
        fixed_z_noise = torch.cat([fixed_z_noise, temp_z_noise], 0)
        temp_y = torch.ones(args["output_num"], 1) + i
        fixed_y = torch.cat([fixed_y, temp_y], 0)

    fixed_z_noise = fixed_z_noise.view(-1, NUM_CLASS * NUM_CLASS, 1, 1)

    # 相当于 onehot 功能
    # onehot = torch.zeros(NUM_CLASS, NUM_CLASS)
    # onehot = onehot.scatter_(1, torch.LongTensor(
    #     np.arange(NUM_CLASS)).view(NUM_CLASS, 1), 1).view(NUM_CLASS, NUM_CLASS, 1, 1)
    fixed_y_label = torch.zeros(NUM_CLASS * NUM_CLASS, NUM_CLASS)
    fixed_y_label.scatter_(1, fixed_y.type(torch.LongTensor), 1)
    fixed_y_label = fixed_y_label.view(-1, NUM_CLASS, 1, 1)

    fixed_z_noise, fixed_y_label = Variable(fixed_z_noise.cuda()), Variable(fixed_y_label.cuda())

    generator.eval()
    gen_imgs = generator(fixed_z_noise, fixed_y_label)
    gen_imgs = Variable(gen_imgs.cpu())
    print("Images generation finished.")
    print(gen_imgs.shape)

    # idx_class = 0
    # i = 1
    # for img in gen_imgs:
    #     if (idx_class == 30):
    #         break
    #     new_img = transform.resize(img[0], (64, 64))
    #     np.savetxt("./trained_models/generator/csv/" + NAME_CLASS[idx_class] + "_" +
    #                str(i) + ".csv", new_img, delimiter = ",")
    #     if (i == 10): 
    #         idx_class += 1
    #         i = 0
    # print("csv generation completed.")

    classifier = resnet34(NUM_CLASS)
    classifier = nn.DataParallel(classifier)
    classifier.load_state_dict(torch.load("./trained_models/classifier/resnet.pth"))
    classifier.cuda()
    print("Classifier model loaded.")

    print("Images classification starting.")
    classifier.eval()

    answer  = []
    imgs_loader = tqdm(gen_imgs, desc = "Classifing")
    for img in enumerate(imgs_loader):
        temp_img = img
        temp_img = transform.resize(temp_img[0], (64, 64))
        plt.imshow(temp_img)

        img = img.view(-1, 1, IMG_SIZE, IMG_SIZE)
        img /= 255.0

        # forward
        output = classifier(img)

        # accuracy
        pred = output.data.max(1)[1]

        # output
        detached = pred.detach().cpu().numpy()
        print(NAME_CLASS[int(detached)] + "\n")
        answer.append(NAME_CLASS[int(detached)])

    print("Images classification finished.")
