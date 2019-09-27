import os
from PIL import Image
import pandas as pd
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file = '', phase='train', input_shape=(3, 112, 112)):
        self.phase = phase
        self.input_shape = input_shape
        self.phase = phase

        imgs = []
        if phase == 'train':
            #把标签转换成字典结构，一个图片对应一个label
            csv_list = pd.read_csv(data_list_file)
            dic = {}

            for _, row in csv_list.iterrows():
                id, pic = row
                imgs.append(pic)

                dic[pic] = int(id)

            self.dic = dic

            imgs = [os.path.join(root, img) for img in imgs]

            self.imgs = np.random.permutation(imgs)

        else:
            self.imgs = [root]


        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.ColorJitter(),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        if self.phase == 'train':
            splits = sample.split('/', 6)[-1]                                 #分割6次，可变
            label = self.dic[splits]
            img_path = sample
            data = Image.open(img_path)
            # data = data.convert('L')
            data = self.transforms(data)
            return data.float(), label
        else:
            data = Image.open(sample)
            data = data.convert('RGB')                        #<-----------------------
            data = self.transforms(data)
            return data

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/mnt/sdb2/liucen/CCF_face/training/',
                      data_list_file='/mnt/sdb2/liucen/CCF_face/training/African_list.csv',
                      phase='train',
                      input_shape=(1, 112, 96))

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

        # cv2.imshow('img', img)
        # cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)