import PIL

from os import listdir
from os.path import isfile, join

import cv2
from PIL import Image
from cv2 import *
import pandas as pd

def resize_images(file_from, path_to, label = True):
    data = pd.read_csv(file_from)
    paths = data["path_img"].tolist()
    if label:
        labels = data["label"].tolist()

    onlyfiles = paths

    allImages = []
    size = 224
    for i, file_path in enumerate(onlyfiles):
        imageOriginal = Image.open(file_path).convert('RGB')
        center_x = imageOriginal.size[0] / 2
        center_y = imageOriginal.size[1] / 2
        if imageOriginal.size[0] > imageOriginal.size[1]:
            imageOriginal = imageOriginal.crop((int(center_x - imageOriginal.size[1] / 2), 0, int(center_x + imageOriginal.size[1] / 2), imageOriginal.size[1]))
        else:
            imageOriginal = imageOriginal.crop((0, int(center_y - imageOriginal.size[0] / 2), imageOriginal.size[0], int(center_y + imageOriginal.size[0] / 2)))

        path = file_path.split("/")
        file_name = path[len(path)-1]
        imageOriginal.save(path_to + "/Cropped_" + file_name)

        image = cv2.imread(path_to + "/Cropped_" + file_name, 1)
        image = cv2.resize(image, (size, size))
        write_path = path_to + "/Resizedd_" + file_name
        if label:
            write_path = path_to + "/" + f"{labels[i]}" + "/Resizedd_" + file_name

        cv2.imwrite(write_path, image)


if __name__ == "__main__":
    print("Resizing images")
    resize_images("train.csv", "resizedTrain")
    print("Resize train finished")
    resize_images("test.csv", "resizedTest", False)
    print("Resize test finished")
    pass
