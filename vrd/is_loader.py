import os, cv2, glob
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from PIL import Image
from sklearn.utils import shuffle
import random
import math
from utils import get_train_img_fullpath_dict, get_image_size
from tqdm import tqdm
import settings


classes_is = [
    '/m/04bcr3', '/m/04dr76w', '/m/01mzpv', '/m/078n6m', '/m/0342h',
    '/m/03m3pdh', '/m/03ssj5', '/m/01_5g', '/m/01y9k5', '/m/0cvnqh',
    '/m/07y_7', '/m/071p9', '/m/05r5c', '/m/01940j', '/m/01s55n',
    '/m/080hkjn', '/m/026t6', '/m/02p5f1q', '/m/0cmx8', '/m/0dt3t',
    '/m/0584n8', '/m/04ctx', '/m/02jvh9'
    ]

classes_is_stoi = { classes_is[i]: i for i in range(len(classes_is)) }

classes = ['none', '/m/083vt', '/m/02gy9n', '/m/05z87', '/m/04lbp', '/m/0dnr7']
stoi = { classes[i]: i for i in range(len(classes)) }

from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, RandomBrightnessContrast,RandomSizedCrop,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue, Resize,CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, VerticalFlip,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, ElasticTransform, ChannelShuffle,RGBShift, Rotate
)

DATA_DIR = settings.DATA_DIR
NUM_CLASSES = 6

class Rotate90(RandomRotate90):
    def apply(self, img, factor=1, **params):
        return np.ascontiguousarray(np.rot90(img, factor))

img_sz = 224

def img_augment(p=1.):
    return Compose([
        #RandomSizedCrop((300, 300), img_sz, img_sz, p=1.),
        Resize(img_sz, img_sz),
        HorizontalFlip(p=0.8),
        #RandomRotate90(p=0.25),
        OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomContrast(),
                RandomBrightness(),
            ], p=0.3),
        #
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=.75 ),
        RandomBrightnessContrast(p=0.2),
        #GaussNoise(),
        #Blur(blur_limit=3, p=.33),
        #OpticalDistortion(p=.33),
        #GridDistortion(p=.33),
        #HueSaturationValue(p=.33)
    ], p=p)

def val_aug(p=1.):
    return Compose([Resize(img_sz, img_sz)], p=1.)


def get_tta_aug(tta_index=0):
    tta_augs = {
        1: [HorizontalFlip(always_apply=True)],
        2: [VerticalFlip(always_apply=True)],
        3: [HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True)],
        4: [Rotate90(always_apply=True)],
        5: [Rotate90(always_apply=True), HorizontalFlip(always_apply=True)],
        6: [VerticalFlip(always_apply=True), Rotate90(always_apply=True)],
        7: [HorizontalFlip(always_apply=True),VerticalFlip(always_apply=True), Rotate90(always_apply=True)],
    }
    if tta_index == 0:
        return Compose([Resize(img_sz, img_sz)], p=1.)
    else:
        return Compose(tta_augs[tta_index], preprocessing_transforms=[Resize(img_sz, img_sz, always_apply=True)], p=1.0)

class ImageDataset(data.Dataset):
    def __init__(self, df, train_mode=True, test_mode=False, tta=None):
        self.df = df
        self.train_mode = train_mode
        self.test_mode = test_mode
        self.tta = tta
        if train_mode:
            self.train_path_dict = get_train_img_fullpath_dict()

    def get_img(self, row):
        if self.train_mode:
            fn = self.train_path_dict[row.ImageID]
        elif self.test_mode:
            fn = os.path.join(settings.TEST_IMG_DIR, row.ImageID+'.jpg')
        else:
            fn = os.path.join(settings.VAL_IMG_DIR, row.ImageID+'.jpg')
        #print(fn)
        full_img = cv2.imread(fn)

        h, w, _ = full_img.shape
        x1, x2, y1, y2 = int(float(row.XMin)*w), math.ceil(float(row.XMax)*w), int(float(row.YMin)*h), math.ceil(float(row.YMax)*h)
        img = full_img[y1:y2, x1:x2, :]
        
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.train_mode:
            aug = img_augment(p=1.)
        elif self.tta is not None:
            aug = get_tta_aug(self.tta)
        else:
            aug = val_aug(p=1.)
        img = aug(image=img)['image']
        
        img = transforms.functional.to_tensor(img)
        img = transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return img

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.get_img(row)    
        if self.test_mode:
            return img, classes_is_stoi[row.LabelName]
        else:
            return img, classes_is_stoi[row.LabelName], stoi[row.LabelName2]

    def __len__(self):
        return len(self.df)

    def collate_fn(self, batch):
        imgs = torch.stack([x[0] for x in batch])
        label1 = torch.tensor([x[1] for x in batch])

        if self.test_mode:
            return imgs, label1
        else:
            labels = torch.tensor([x[2] for x in batch])
            return imgs, label1, labels

def prepare_dataset(box_file='challenge-2019-train-vrd-bbox.csv', vrd_file='challenge-2019-train-vrd.csv'):
    df_vrd = pd.read_csv(os.path.join(DATA_DIR, vrd_file), converters={
        'XMin1':lambda x: '{:.6f}'.format(float(x)),
        'XMax1':lambda x: '{:.6f}'.format(float(x)),
        'YMin1':lambda x: '{:.6f}'.format(float(x)),
        'YMax1':lambda x: '{:.6f}'.format(float(x))})
    print(df_vrd.shape)
    df_vrd_is = df_vrd.loc[df_vrd.RelationshipLabel=='is'].copy()
    df_vrd_is['box_id'] = df_vrd_is.ImageID + '_' + df_vrd_is.LabelName1 + '_' + df_vrd_is.XMin1 + '_' + df_vrd_is.XMax1 + '_' + df_vrd_is.YMin1 + '_' + df_vrd_is.YMax1
    print(df_vrd_is.shape)
    
    #df_box = pd.read_csv(os.path.join(DATA_DIR, box_file), dtype={'XMin':str, 'XMax':str, 'YMin': str, 'YMax': str})
    df_box = pd.read_csv(os.path.join(DATA_DIR, box_file), converters={
        'XMin':lambda x: '{:.6f}'.format(float(x)),
        'XMax':lambda x: '{:.6f}'.format(float(x)),
        'YMin':lambda x: '{:.6f}'.format(float(x)),
        'YMax':lambda x: '{:.6f}'.format(float(x))})
    print(df_box.shape)
    is_labels_1 = df_vrd.loc[df_vrd.RelationshipLabel=='is'].LabelName1.unique()
    print('is_label_1:', len(is_labels_1))
    df_box_is = df_box.loc[df_box.LabelName.isin(set(is_labels_1))].copy()
    print(df_box_is.shape)
    df_box_is['box_id'] = df_box_is.ImageID + '_' + df_box_is.LabelName + '_' + df_box_is.XMin + '_' + df_box_is.XMax + '_' + df_box_is.YMin + '_' + df_box_is.YMax


    df_train_is = df_box_is.set_index('box_id').join(df_vrd_is.set_index('box_id'), on='box_id', rsuffix='dfvrd')
    #print(df_train_is.sample(10))
    #print(df_train_is.columns)
    print(df_train_is.shape)
    df_train_is.LabelName2 = df_train_is.LabelName2.fillna('none')
    print(df_train_is.LabelName2.value_counts())

    print(df_train_is.head())
    #print(df_train_is.columns)
    #print(df_train_is.LabelName.value_counts())
    #print(df_train_is.loc[df_train_is.LabelName == df_train_is.LabelName1].shape)
    #print(df_train_is.loc[df_train_is.LabelName != df_train_is.LabelName1].shape)
    #print(df_train_is.shape)
    return df_train_is

def get_train_loader(batch_size=4, dev_mode=False):
    df = prepare_dataset()
    if dev_mode:
        df = df.iloc[:100]
    ds = ImageDataset(df, train_mode=True, test_mode=False)
    train_loader = data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=24, collate_fn=ds.collate_fn, drop_last=True)
    train_loader.num = len(ds)
    return train_loader


def get_val_loader(batch_size=4, dev_mode=False, val_num=10000):
    df = prepare_dataset(box_file='challenge-2019-validation-vrd-bbox.csv', vrd_file='challenge-2019-validation-vrd.csv')
    if dev_mode:
        df = df.iloc[:100]
    df = df.iloc[:val_num]
    print('data len:', len(df))
    ds = ImageDataset(df, train_mode=False, test_mode=False)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=24, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(ds)
    return loader


def get_det(pred_str):
    dets = []
    det = []
    for i, e in enumerate(pred_str.split(' ')):
        if i % 6 == 0:
            det = []
        det.append(e)
        if (i+1) % 6 == 0:
            if det[0] in set(classes_is): #and float(det[1]) > 0.1:
                dets.append(det)

    return dets

def get_test_loader(dets_file_name, batch_size=4, dev_mode=False):
    print('loading: {} ...'.format(dets_file_name))
    df_det = pd.read_csv(dets_file_name)
    df_det['dets'] = df_det.PredictionString.map(lambda x: get_det(str(x)))

    img_ids, labels, confs, xmins, ymins, xmaxs, ymaxs = [], [], [], [], [], [], []
    for img_id, dets in tqdm(zip(df_det.ImageId.values, df_det.dets.values)):
        for det in dets:
            if float(det[4]) - float(det[2]) > 0.005 and float(det[5]) - float(det[3]) > 0.005:
                img_ids.append(img_id)
                labels.append(det[0])
                confs.append(det[1])
                xmins.append(det[2])
                ymins.append(det[3])
                xmaxs.append(det[4])
                ymaxs.append(det[5])
    
    df_test = pd.DataFrame({
        'ImageID': img_ids,
        'LabelName': labels,
        'Confidence': confs,
        'XMin': xmins,
        'YMin': ymins,
        'XMax': xmaxs,
        'YMax': ymaxs
    })
    if dev_mode:
        df_test = df_test.iloc[:100]

    print('DATA LEN', df_test.shape)

    ds = ImageDataset(df_test, train_mode=False, test_mode=True)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=24, collate_fn=ds.collate_fn, drop_last=False)
    loader.num = len(ds)
    loader.df = df_test

    return loader


def test_train_loader():
    train_loader = get_train_loader(dev_mode=False)
    for img, label1, labels in train_loader:
        print(img.size())
        print(label)
        break

def test_test_loader():
    test_loader = get_test_loader('/mnt/chicm/open-images-vrd/notebooks/sub_detect_0724.csv', 4, dev_mode=True)
    for img, label1 in test_loader:
        print(img.size())
        break

if __name__ == '__main__':
    test_train_loader()
    #test_test_loader()
    #test_index_loader()
