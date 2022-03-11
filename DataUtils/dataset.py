import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

class QuickdrawDataset(Dataset):
    def __init__(self, type, input_dir = "./Data/dataset/", num_class = 10, img_size = 32):
        data, target = self.get_dataset(input_dir, type)

        data = torch.from_numpy(data)
        data = data / 255.0

        img_h, img_w = int(math.sqrt(data.shape[1])), int(math.sqrt(data.shape[1]))
        data = np.reshape(data, [data.shape[0], 1, img_h, img_w])

        padding = int((img_size - img_w) / 2)
        data = np.pad(data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                      "constant", constant_values = (0, 0))

        self.image = data
        self.label = target
        self.num_class = num_class
        print("data shape: " + str(self.image.shape))
        print("data shape: " + str(self.label.shape))
        print("Dataset " + type + " loading done.")
        print("*" * 50 + "\n")

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        return image, label

    def __len__(self):
        return len(self.image)

    def get_num_class(self):
        return self.num_class

    def get_dataset(self, input_dir, type):
        if os.path.exists(input_dir + type + ".npz"):
            print("*" * 50)
            print("Loading " + type + " dataset...")
            print("*" * 50)
            data_cache = np.load(input_dir + type + ".npz")
            return data_cache["data"].astype("float32"), data_cache["target"].astype("int64")

        else:
            raise FileNotFoundError("%s doesn't exist!" % input_dir + type + ".npz")
