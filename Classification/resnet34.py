import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import torch
from tqdm import tqdm

import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models import resnet34, QuickDrawNet
from DataUtils.config import *
from DataUtils.dataset import QuickdrawDataset

if __name__ == "__main__":
    if not os.path.isdir("./trained_models/"):
        os.mkdir("./trained_models/")
    if not os.path.isdir("./trained_models/classifier/"):
        os.mkdir("./trained_models/classifier/")

    print("*" * 50)
    print("Loading the data...")
    train_data = QuickdrawDataset(type = "train", input_dir = "./Data/dataset/", num_class = NUM_CLASS, img_size = IMG_SIZE)
    train_loader = DataLoader(train_data, batch_size = TRAIN_BATCH_SIZE, shuffle = True)

    test_data = QuickdrawDataset(type = "test", input_dir = "./Data/dataset/", num_class = NUM_CLASS, img_size = IMG_SIZE)
    test_loader = DataLoader(test_data, batch_size = TEST_BATCH_SIZE, shuffle = True)

    print("Train images number: %d" % len(train_data))
    print("Test images number: %d" % len(test_data))

    # network
    net = resnet34(NUM_CLASS)
    net = nn.DataParallel(net)
    net.cuda()

    net_opt = SGD(net.parameters(), lr = RESNET_LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)

    train_accuracy = []
    train_loss = []
    test_accuracy = []
    test_loss = []

    def train():
        correct = 0
        loss_avg = 0.0

        net.train()

        data_loader = tqdm(train_loader, desc = "Training")

        for idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            data = data.view(-1, 1, IMG_SIZE, IMG_SIZE)

            # forward
            output = net(data)

            # backward
            net_opt.zero_grad()
            loss = F.cross_entropy(output, target)
            loss.backward()
            net_opt.step()

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # exponential moving average
            loss_avg = loss_avg * 0.2 + float(loss) * 0.8

        train_accuracy.append(correct / len(train_loader.dataset))
        train_loss.append(loss_avg)

    def test():
        correct = 0
        loss_avg = 0.0

        net.eval()

        data_loader = tqdm(test_loader, desc = "Testing")

        for idx, (data, target) in enumerate(data_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())

            data = data.view(-1, 1, IMG_SIZE, IMG_SIZE)

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += float(pred.eq(target.data).sum())

            # test loss average
            loss_avg += float(loss)

        test_accuracy.append(correct / len(test_loader.dataset))
        test_loss.append(loss_avg / len(test_loader))

    # Main loop
    best_accuracy = 0.0
    for epoch in range(TRAIN_EPOCH):
        print("*" * 50)
        print("epoch " + str(epoch + 1) + " is running...")
        if epoch + 1 in LR_DECAY_STEP:
            lr = RESNET_LR * GAMMA
            for param_group in net_opt.param_groups:
                param_group["lr"] = lr

        train()
        print("")
        test()
        print("")

        if max(test_accuracy) > best_accuracy:
            best_accuracy = max(test_accuracy)
            torch.save(net.state_dict(), "./trained_models/classifier/resnet.pth")

        print("Best accuracy: %.4f" % best_accuracy)
        print("*" * 50)
