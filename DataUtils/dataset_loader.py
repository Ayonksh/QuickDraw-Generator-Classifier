import os
import glob
import requests
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import randint

from config import *

def download_data(categories, output_dir = "./Data/rawdata/"):
    print("Download starting")

    for i in tqdm(range(len(categories))):
        cat = categories[i]
        cat_name = cat.replace(" ", "%20")
        url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy".format(cat_name)
        res = requests.get(
            url,
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
                "Connection": "close"
            },
            allow_redirects = True,
            verify = False
        )
        open(output_dir + "{}.npy".format(cat), "wb").write(res.content)

    print("Download completed")

def generate_dataset(input_dir = "./Data/rawdata/", output_dir = "./Data/dataset/", max_samples_per_class = 50000, show_imgs = True):
    print("*" * 50)
    print("Generate dataset from npy data")
    print("*" * 50)
    all_files = glob.glob(input_dir +  "*.npy")
    print("Classes number: " + str(len(all_files)))
    print("~" * 50)

    image = np.empty([0, 28 * 28]) # the size of image from .npy file is 28
    label = np.empty([0])
    class_names = []
    class_samples_num = []

    for idx, file in enumerate(all_files):
        data = np.load(file)

        indices = np.arange(0, data.shape[0])
        # randomly choose max_items_per_class data from each class
        indices = np.random.choice(indices, max_samples_per_class, replace = False)
        data = data[indices]

        image = np.concatenate((image, data), axis = 0)

        temp_label = np.full(data.shape[0], idx)
        label = np.append(label, temp_label)

        file_name, ext = os.path.splitext(os.path.basename(file))
        class_name = file_name[18:]
        class_names.append(class_name)
        class_samples_num.append(str(data.shape[0]))

        print(str(idx + 1) + "/" + str(len(all_files)) + " - " + class_name +
              " has been loaded. \n\t Totally " + str(data.shape[0]) + " samples.")
        print("~" * 50)

    print("\n" + "*" * 50)
    print("Data loading done.")
    print("*" * 50)
    data = None
    temp_label = None

    # randomize the dataset
    permutation = np.random.permutation(label.shape[0])
    image = image[permutation, :]
    label = label[permutation]

    # separate into training and testing
    test_size = int(image.shape[0] * 0.2)
    image_test = image[0:test_size, :]
    label_test = label[0:test_size]
    image_train = image[test_size:image.shape[0], :]
    label_train = label[test_size:label.shape[0]]

    print("image_train size: ")
    print(image_train.shape)
    print("\nimage_test size: ")
    print(image_test.shape)

    if show_imgs:
        plt.figure("random images from dataset", figsize = (9, 9))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            idx = randint(0, len(image_train) - 1)
            plt.imshow(image_train[idx].reshape(28, 28))
            plt.title(class_names[int(label_train[idx])])
            plt.axis("off")
        plt.show()

    np.savez_compressed(output_dir + "train", data = image_train, target = label_train)
    np.savez_compressed(output_dir + "test", data = image_test, target = label_test)

    print("*" * 50)
    print("Great, data_cache has been saved into disk.")
    print("*" * 50)

if __name__ == "__main__":
    if not os.path.isdir("./Data/"):
        os.mkdir("./Data/")
    if not os.path.isdir("./Data/rawdata/"):
        os.mkdir("./Data/rawdata/")
    if not os.path.isdir("./Data/dataset/"):
        os.mkdir("./Data/dataset/")

    download_data(NAME_CLASS)
    generate_dataset(max_samples_per_class = SAMPLE_SIZE, show_imgs = False)
