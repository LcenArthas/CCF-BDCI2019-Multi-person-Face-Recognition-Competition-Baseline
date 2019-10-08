import os
from PIL import Image
from imgaug import  augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys

class Dataset(data.Dataset):
    def __init__(self, root, phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape
        self.phase = phase

        imgs = []
        if phase == 'train':

            for pic in os.listdir(root):
                imgs.append(pic)

            imgs = [os.path.join(root, img) for img in imgs]
            self.imgs = np.random.permutation(imgs)

        else:
            self.imgs = [root]

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(self.input_shape[1:]),
                # T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        if self.phase == 'train':
            splits = sample.split('/')[-1]               
            label = int(splits.split('_')[0])
            img_path = sample
            data = Image.open(img_path).convert('RGB')
            data = self.transforms(data)
            return data, label
        else:
            data = Image.open(sample)
            data = data.convert('RGB')
            data = self.transforms(data)
            return data

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='../train_data/new_training/train/',
                      phase='train',
                      input_shape=(1, 112, 112))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        plt.imshow(img)
        plt.show()

        # cv2.imshow('img', img)
        # cv2.waitKey()
        break
        # dst.decode_segmap(labels.numpy()[0], plot=True)