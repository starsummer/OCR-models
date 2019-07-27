import lmdb
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import six
from PIL import Image
import numpy as np
import re
import math
import random
import cv2
import os
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose,ElasticTransform,RGBShift,
    InvertImg,JpegCompression,Cutout,RandomScale,RandomSunFlare,RandomShadow,DualIAATransform,
    IAAPiecewiseAffine,ChannelShuffle
)


def strong_aug(p=0.8):
    return Compose([
        # RandomRotate90(),
        # Flip(),
        # Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([ #模糊
            MotionBlur(p=0.5),
            MedianBlur(blur_limit=3, p=0.5),
            Blur(blur_limit=3, p=0.5),
            JpegCompression(p=1,quality_lower=7,quality_upper=40)
        ], p=1),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([ 
            IAAPiecewiseAffine(p=1,scale=(0.005, 0.01), nb_rows=4, nb_cols=4),
            IAAPerspective(p=1,scale=(random.uniform(0.01,0.03),random.uniform(0.01, 0.03))),
            ElasticTransform(p=1,alpha=random.randint(50,100), sigma=random.randint(8,13), alpha_affine=0,border_mode=3),
        ], p=0.2),
        OneOf([ 
            ElasticTransform(p=1,alpha=random.randint(50,100), sigma=random.randint(8,13), alpha_affine=0,border_mode=3),
        ], p=0.6),
        OneOf([ 
            OpticalDistortion(p=1,distort_limit=0.2,border_mode=3),
            # GridDistortion(p=1,distort_limit=0.1,border_mode=3),
        ], p=0.1),        
        OneOf([
            CLAHE(clip_limit=2,p=0.5),
            # IAASharpen(),
            IAAEmboss(p=0.5),
            RandomBrightnessContrast(p=1), #随机调整亮度饱和度，和下一个区别？
            HueSaturationValue(p=1), #随机调整hsv值
            RGBShift(p=0.5), #随机调整rgb值
            ChannelShuffle(p=0.5), #RGB通道调换
            InvertImg(p=0.1), #255-像素值，反转图像
        ], p=0.5),    

    ], p=p) 


class lmdbDataLoader(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, char2index=dict() ):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.char2index = char2index
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)
                # img = Image.open(buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = re.sub(r'\s+',' ',label)
            # label = ''.join(label[i] if label[i] in self.char2index else '' 
            #     for i in range(len(label)))
            clean_label = ''
            for char in label:
                if char not in self.char2index:
                    print(char + ' not in dict')
                    continue
                clean_label += char
            label = clean_label

            if len(label) <= 0:
                return self[index + 1]
            if self.reverse:
                label_rev = label[-1::-1]
                # label_rev += '$'
            # label += '$'
            
            if self.transform is not None:
                img = self.transform(img)

        if self.reverse:
            return (img, label, label_rev)
        else:
            return (img, label)

class loader(object):
    def __init__(self, dataset,data_cfg,shuffle=True,num_workers=4,drop_last=False,test=False,aug=strong_aug(p=1)):
        super(loader, self).__init__()
        self.dataset = dataset
        self.nc = data_cfg['nc']
        if test:
            self.batch_size = data_cfg['test_batch_size']
        else:
            self.batch_size = data_cfg['batch_size']
        self.target_H = data_cfg['target_H']
        self.test = test
        self.max_width = data_cfg['max_width']
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.aug = aug
        self.is_aug = data_cfg['is_aug']
        self.fix_width = data_cfg['fix_width']
        self.transform_tensor=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.interpolation_arr = [Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.LANCZOS]
 

    def aug_resize_transform(self, img):   

        #进行增强  
        if self.test==False and self.is_aug == True:
            if random.randint(0,100)>50: #进行一定缩放，一是改变字符宽高比，二是模拟压缩失真
                img_W, img_H = img.size
                delta_width = random.uniform(0.8,1.2) #宽高随机为原来的0.8-1.2倍
                W = math.ceil(delta_width*img_W)
                delta_height = random.uniform(0.8,1.2)
                H = math.ceil(delta_height*img_H)
                img = img.resize((W,H), random.choice([Image.NEAREST,Image.BILINEAR,Image.BICUBIC,Image.LANCZOS])) #随机选择4种插值方式        

            if random.randint(0,100)>75: #减少像素
                img_W, img_H = img.size
                random_h = random.randint(13,40)
                new_W = int(img_W * random_h / img_H)
                img = img.resize((new_W,random_h), Image.BICUBIC)      

            img = np.asarray(img)
            img = self.aug(image=img)['image']
            img = Image.fromarray(img)

        #resize到固定高度
        img_W, img_H = img.size
        new_W = int(img_W * self.target_H / img_H)
        if new_W>self.max_width: #长度最高为max_width
            new_W = self.max_width
        # print(new_W)
        img = img.resize((new_W,self.target_H), Image.BICUBIC)    
        return img   

    def pad_same_width(self, img, batch_max_width):
        # pad_width = batch_max_width - img.size[0]
        new_image = Image.new('RGB', (batch_max_width, self.target_H) )
        new_image.paste(img,(0,0))
        # name = random.randint(1,10000000)
        # new_image.save('./test/{}_old.jpg'.format(name)) 
        return new_image

    def collate_fn(self, batch):
        #图像增强并resize到相同高度
        imgs = [self.aug_resize_transform(x[0]) for x in batch]
        labels = [ x[1] for x in batch]    

        #pad到这一batch最大长度    
        batch_max_width = max(im.size[0] for im in imgs)
        if self.fix_width: #如果宽度固定，直接resize到max_width
            batch_max_width = self.max_width
        # print(f'batch_max_width:{batch_max_width}')
        imgs = [self.pad_same_width(img,batch_max_width) for img in imgs]    

        if self.nc == 1:
            imgs = [img.convert('L') for img in imgs]
        #转化为tensor并正则化
        imgs = [self.transform_tensor(img) for img in imgs]
        imgs = torch.stack(imgs,0)
        return imgs,labels

    def main(self):
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers,
            collate_fn = self.collate_fn,
            drop_last = self.drop_last
        )
        return train_loader

def get_train_loader(data_cfg,char2index):
    train_lmdb = data_cfg['train_lmdb']
    if isinstance(train_lmdb,str): #是文件路径，放到list里面
        train_lmdb_list = [train_lmdb]
    elif isinstance(train_lmdb,list):
        train_lmdb_list = train_lmdb

    train_data_list = []
    for lmdb_path in train_lmdb_list:
        train_data = lmdbDataLoader(root=lmdb_path,char2index=char2index)
        train_data_list.append(train_data)
    print(train_data_list)
    train_data_total = torch.utils.data.ConcatDataset(train_data_list)
    train_loader = loader(train_data_total,data_cfg).main()
    return train_loader


def get_test_loader(data_cfg,char2index):
    test_datasets = [os.path.join(item) for item in data_cfg['test_lmdb']]
    test_loader_dict = dict() #将多个测试集放到字典里
    for item in test_datasets:
        item = os.path.join(data_cfg['test_root'],item)
        test_name = os.path.basename(item)
        print(test_name)
        test_dataset =  lmdbDataLoader(root=item,char2index=char2index)
        test_loader = loader(test_dataset,data_cfg,test=True).main()
        test_loader_dict[test_name] = test_loader
    return test_loader_dict



if __name__ == '__main__':

    train_loader = loader(train_data).get_loader()
    for i in range(1):    
        train_iter = iter(train_loader)
        images,text = train_iter.next() 
        print(type(images))
        print(images.size())
        print(text)

