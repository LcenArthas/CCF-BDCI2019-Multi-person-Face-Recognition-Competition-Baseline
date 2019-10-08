#这个脚本的目的在于对原始的数据集进行处理，划分验证集

import os
import random
import shutil
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

root_path = "../train_data/"

#生成新的文件夹存储划分的数据
if os.path.exists(root_path + 'new_training'):
    pass
else:
    os.makedirs(root_path + 'new_training')

if os.path.exists(root_path + 'new_training/train'):
    pass
else:
    os.makedirs(root_path + 'new_training/train')

#============================================================================================
race_list = ['African', 'Caucasian', 'Asian', 'Indian']
id = 1                                                         #对每个人种图片编写id
for race in race_list:
    race_satlist = []                                          #每个race中图片个数统计
    data_path = root_path + 'training/' + race
    data_list = os.listdir(data_path)
    data_len = len(data_list)                                  #每个人种的人数
    print(race, '有:', data_len)

    if race == 'Caucasian':
        val_num = int(data_len * 0.0)
    else:
        val_num = int(data_len * 0.0)

    val_list = random.sample(data_list, val_num)
    train_list = list(set(data_list).difference(set(val_list)))

    #复制制作训练集
    for i in train_list:
        pic_file = data_path + '/' + i
        race_satlist.append(len(os.listdir(pic_file)))
        for ind, j in enumerate(os.listdir(pic_file)):
            pic = pic_file + '/' + j                            #精确到图片
            new_pic = root_path + 'new_training/train/' + str(id) + '_' + str(ind) + '.jpg'
            shutil.copyfile(pic, new_pic)                       #复制图片
        id += 1
