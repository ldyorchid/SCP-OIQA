import os
import numpy as np
import pandas as pd
#import cv2
import torch
from torch.utils.data.dataset import Dataset

from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
transformations_1 = transforms.Compose([transforms.Resize(224),  transforms.ToTensor()])

transformations_2 = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=0.485, std=0.229)])


class Dataset(Dataset):
    '''
        csv——path : csv文件存储路径
        sep:指定分割符形式，CSV默认逗号分割，可以忽略这个参数，如果是其它分割方式，则要填写
        names: 指定列名，通常用一个字符串列表表示，没有header参数时，用names会增加一行作为列名，原数据的第一行仍然保留
        index_col: 一个字符串列表，指定哪几列作为索引
        encoding="utf-8-sig"  ：feff是一个BOM(Byte Order Mark)，是一个不显示的标识字段，在utf-16或者utf-32等中，feff放在首位表示字节流高位在前还是低位在前；
                                但是一般的utf-8是不需要BOM的，为了解决这个问题，我们采用utf-8-sig编码打开csv文件，可以看到已经正常了
    '''
    def __init__(self, data_dir, csv_path, transform, test = False):
        column_names = ['1','1_rbd','2','2_rbd','3','3_rbd','4','4_rbd','5','5_rbd','6','6_rbd','MOS']
        tmp_df = pd.read_csv(csv_path, sep=',', names=column_names, index_col=False, encoding="utf-8-sig")
        self.data_dir = data_dir
        self.transform = transform
        self.X_train = tmp_df[['1','1_rbd','2','2_rbd','3','3_rbd','4','4_rbd','5','5_rbd','6','6_rbd']]  #给每一列分类一个标签
        self.Y_train = tmp_df['MOS']
        self.data_dir1 = "F:\OIQA\saliency"#'F:\OIQA\saliency'  F:\saliency\saliency
        self.length = len(tmp_df)
        self.test = test

    def __getitem__(self, index):

        img_group = []
        sa_group = []
        path_1 = os.path.join(self.data_dir,self.X_train.iloc[index,0])  #index是行，0是列
        path_1_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 1])
        path_2 = os.path.join(self.data_dir,self.X_train.iloc[index,2])
        path_2_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 3])
        path_3 = os.path.join(self.data_dir,self.X_train.iloc[index, 4])
        path_3_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 5])
        path_4 = os.path.join(self.data_dir,self.X_train.iloc[index, 6])
        path_4_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 7])
        path_5 = os.path.join(self.data_dir,self.X_train.iloc[index, 8])
        path_5_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 9])
        path_6 = os.path.join(self.data_dir,self.X_train.iloc[index, 10])
        path_6_rbd = os.path.join(self.data_dir1, self.X_train.iloc[index, 11])


        img_1 = Image.open(path_1)
        img_1 = img_1.convert('RGB')
        img_1_rbd = Image.open(path_1_rbd)

        img_2 = Image.open(path_2)
        img_2 = img_2.convert('RGB')
        img_2_rbd = Image.open(path_2_rbd)

        img_3 = Image.open(path_3)
        img_3 = img_3.convert('RGB')
        img_3_rbd = Image.open(path_3_rbd)

        img_4 = Image.open(path_4)
        img_4 = img_4.convert('RGB')
        img_4_rbd = Image.open(path_4_rbd)

        img_5 = Image.open(path_5)
        img_5 = img_5.convert('RGB')
        img_5_rbd = Image.open(path_5_rbd)

        img_6 = Image.open(path_6)
        img_6 = img_6.convert('RGB')
        img_6_rbd = Image.open(path_6_rbd)

        img_1 = self.transform(img_1)
        img_group.append(img_1.numpy())
        img_1_rbd = transformations_1(img_1_rbd)
        sa_group.append(img_1_rbd.numpy())

        img_2 = self.transform(img_2)
        img_group.append(img_2.numpy())
        img_2_rbd = transformations_1(img_2_rbd)
        sa_group.append(img_2_rbd.numpy())

        img_3 = self.transform(img_3)
        img_group.append(img_3.numpy())
        img_3_rbd = transformations_1(img_3_rbd)
        sa_group.append(img_3_rbd.numpy())

        img_4 = self.transform(img_4)
        img_group.append(img_4.numpy())
        img_4_rbd = transformations_1(img_4_rbd)
        sa_group.append(img_4_rbd.numpy())

        img_5 = self.transform(img_5)
        img_group.append(img_5.numpy())
        img_5_rbd = transformations_1(img_5_rbd)
        sa_group.append(img_5_rbd.numpy())

        img_6 = self.transform(img_6)
        img_group.append(img_6.numpy())
        img_6_rbd = transformations_1(img_6_rbd)
        sa_group.append(img_6_rbd.numpy())

        img_group = np.array(img_group)
        img_group = torch.from_numpy(img_group)
        sa_group = np.array(sa_group)
        sa_group = torch.from_numpy(sa_group)
        y_mos = self.Y_train.iloc[index]


        y_label = torch.FloatTensor(np.array(float(y_mos)))


        return img_group, sa_group, y_label


    def __len__(self):
        return self.length






