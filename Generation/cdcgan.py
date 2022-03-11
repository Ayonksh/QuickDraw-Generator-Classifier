import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import time
import pickle
import imageio
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import BCELoss
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import CDCGAN_Generator, CDCGAN_Discriminator
from DataUtils.config import *
from DataUtils.dataset import QuickdrawDataset

def show_result(idx_epoch, num_class, show = False, save = False, path = "result.png"):
    # fixed noise & label
    fixed_z_noise = torch.randn(NUM_CLASS, NUM_CLASS * NUM_CLASS)
    fixed_y = torch.zeros(NUM_CLASS, 1)
    for i in range(NUM_CLASS - 1):
        temp_z_noise = torch.randn(NUM_CLASS, NUM_CLASS * NUM_CLASS)
        fixed_z_noise = torch.cat([fixed_z_noise, temp_z_noise], 0)
        temp_y = torch.ones(NUM_CLASS, 1) + i
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

    gen.eval()
    gen_imgs = gen(fixed_z_noise, fixed_y_label)
    gen.train()

    size_figure_grid = num_class
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize = (5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(num_class * num_class):
        i = k // num_class
        j = k % num_class
        ax[i, j].cla()
        ax[i, j].imshow(gen_imgs[k, 0].cpu().data.numpy(), cmap = "gray")

    label = "Epoch {}".format(idx_epoch)
    fig.text(0.5, 0.04, label, ha = "center")
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path="train_hist.png"):
    x = range(len(hist["disc_losses"]))

    y1 = hist["disc_losses"]
    y2 = hist["gen_losses"]

    plt.plot(x, y1, label = "disc_loss")
    plt.plot(x, y2, label = "gen_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend(loc = 4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    if not os.path.isdir("./trained_models/"):
        os.mkdir("./trained_models/")
    if not os.path.isdir("./trained_models/generator/"):
        os.mkdir("./trained_models/generator/")
    if not os.path.isdir("./trained_models/generator/fixed_results/"):
        os.mkdir("./trained_models/generator/fixed_results/")

    train_data = QuickdrawDataset(type = "train", input_dir = "./Data/dataset/", num_class = NUM_CLASS, img_size = IMG_SIZE)
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)

    # network
    gen = CDCGAN_Generator(NUM_CLASS, DEPTH)
    disc = CDCGAN_Discriminator(NUM_CLASS, DEPTH)
    gen.init_weight(mean = 0.0, std = 0.02)
    disc.init_weight(mean = 0.0, std = 0.02)
    gen.cuda()
    disc.cuda()

    gen_opt = Adam(gen.parameters(), lr = CDCGAN_LR, betas = (0.5, 0.999), weight_decay = CDCGAN_LR / TRAIN_EPOCH)
    disc_opt = Adam(disc.parameters(), lr = CDCGAN_LR, betas = (0.5, 0.999), weight_decay = CDCGAN_LR / TRAIN_EPOCH)

    train_hist = {}
    train_hist["disc_losses"] = []
    train_hist["gen_losses"] = []
    train_hist["per_epoch_time"] = []
    train_hist["total_time"] = []

    # label preprocess
    onehot = torch.zeros(NUM_CLASS, NUM_CLASS)
    onehot = onehot.scatter_(1, torch.LongTensor(
        np.arange(NUM_CLASS)).view(NUM_CLASS, 1), 1).view(NUM_CLASS, NUM_CLASS, 1, 1)
    mask = torch.zeros([NUM_CLASS, NUM_CLASS, IMG_SIZE, IMG_SIZE])
    for i in range(NUM_CLASS):
        mask[i, i, :, :] = 1

    # Binary Cross Entropy loss
    BCE_loss = BCELoss()

    print("training start!")
    start_time = time.time()

    for epoch in range(TRAIN_EPOCH):
        disc_losses = []
        gen_losses = []

        epoch_start_time = time.time()

        y_real = torch.ones(BATCH_SIZE)
        y_fake = torch.zeros(BATCH_SIZE)
        y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())

        for i, (x_image, y_label) in enumerate(train_loader):

            # show the image
            # print(image.size())
            # plt.figure("9 images from dataset", figsize=(9, 9))
            # for i in range(9):
            #     plt.subplot(3, 3, i + 1)
            #     plt.title(NAME_CLASS[int(label[i])])
            #     plt.imshow(image[i].permute(1, 2, 0))
            #     plt.axis("off")
            # plt.show()

            ##############################
            #   Training discriminator   #
            ##############################

            mini_batch = x_image.size()[0]
            if mini_batch != BATCH_SIZE:
                y_real = torch.ones(mini_batch)
                y_fake = torch.zeros(mini_batch)
                y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())

            # 1.) Training with real
            disc.zero_grad()

            x_image = x_image.float()
            y_label = y_label.long()
            y_label_mask = mask[y_label]
            x_image, y_label_mask = Variable(x_image.cuda()), Variable(y_label_mask.cuda())

            disc_result = disc(x_image, y_label_mask).squeeze()
            disc_real_loss = BCE_loss(disc_result, y_real)

            # 2.) Train with fake
            z_noise = torch.randn((mini_batch, NUM_CLASS * NUM_CLASS)).view(-1, NUM_CLASS * NUM_CLASS, 1, 1)
            y_label = (torch.rand(mini_batch, 1) * NUM_CLASS).type(torch.LongTensor).squeeze()
            y_label_onehot = onehot[y_label]
            y_label_mask = mask[y_label]
            z_noise, y_label_onehot, y_label_mask = Variable(
                z_noise.cuda()), Variable(y_label_onehot.cuda()), Variable(y_label_mask.cuda())

            gen_result = gen(z_noise, y_label_onehot)
            disc_result = disc(gen_result, y_label_mask).squeeze()

            disc_fake_loss = BCE_loss(disc_result, y_fake)
            # disc_fake_score = disc_result.data.mean()

            disc_train_loss = disc_real_loss + disc_fake_loss

            disc_opt.zero_grad()
            disc_train_loss.backward()
            disc_opt.step()

            disc_losses.append(disc_train_loss.data)

            ##########################
            #   Training generator   #
            ##########################
            gen.zero_grad()

            gen_result = gen(z_noise, y_label_onehot)
            disc_result = disc(gen_result, y_label_mask).squeeze()

            # IMPORTANT SETP
            # calculate the loss based on the predictions given by the Discriminator
            # on the images generated by the Generator
            gen_train_loss = BCE_loss(disc_result, y_real)

            gen_opt.zero_grad()

            gen_train_loss.backward()
            gen_opt.step()

            gen_losses.append(gen_train_loss.data)

            print("epoch [%d/%d] batch [%d/%d] - loss_disc: %.3f, loss_gen: %.3f"
                  % ((epoch + 1), TRAIN_EPOCH, i, len(train_loader),
                  torch.mean(torch.FloatTensor(disc_losses)),
                  torch.mean(torch.FloatTensor(gen_losses))))

        epoch_end_time = time.time()
        per_epoch_time = epoch_end_time - epoch_start_time
        print("[%d/%d] - per_epoch_time: %.2f, loss_disc: %.3f, loss_gen: %.3f"
              % ((epoch + 1), TRAIN_EPOCH, per_epoch_time, 
              torch.mean(torch.FloatTensor(disc_losses)),
              torch.mean(torch.FloatTensor(gen_losses))))

        fixed_path = "./trained_models/generator/fixed_results/epoch-" + str(epoch + 1) + ".png"
        show_result((epoch + 1), NUM_CLASS, save = True, path = fixed_path)

        train_hist["disc_losses"].append(torch.mean(torch.FloatTensor(disc_losses)))
        train_hist["gen_losses"].append(torch.mean(torch.FloatTensor(gen_losses)))
        train_hist["per_epoch_ptimes"].append(per_epoch_time)

    end_time = time.time()
    total_time = end_time - start_time
    train_hist["total_time"].append(total_time)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f"
          % (torch.mean(torch.FloatTensor(train_hist["per_epoch_times"])), TRAIN_EPOCH, total_time))

    print("Training finish!... save training results")
    with open("./trained_models/generator/train_hist.pkl", "wb") as f:
        pickle.dump(train_hist, f)

    show_train_hist(train_hist, save = True, path = "./trained_models/generator/train_hist.png")

    torch.save(gen.state_dict(), "./trained_models/generator/generator.pth")
    torch.save(disc.state_dict(), "./trained_models/generator/discriminator.pth")

    images = []
    for epoch in range(TRAIN_EPOCH):
        img_name = "./trained_models/generator/fixed_results/epoch-" + str(epoch + 1) + ".png"
        images.append(imageio.imread(img_name))
    imageio.mimsave("./trained_models/generator/generation_animation.gif", images, fps = 5)
