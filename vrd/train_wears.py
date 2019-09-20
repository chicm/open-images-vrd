import os
import argparse
import numpy as np
import pandas as pd
import random
import torch
import json
import cv2
from tqdm import tqdm
import pickle
import glob
import os.path as osp
from multiprocessing import Pool
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from utils import get_iou, get_image_size

import settings
DATA_DIR = settings.DATA_DIR

classes_1 = set(['/m/03bt1vf', '/m/04yx4', '/m/05r655', '/m/01bl7v'])
classes_2 = set(['/m/080hkjn', '/m/071p9', '/m/01940j', '/m/06__v', '/m/0584n8', '/m/01s55n'])

classes = set([
    '/m/03bt1vf,/m/080hkjn', '/m/04yx4,/m/071p9', '/m/04yx4,/m/01940j',
    '/m/04yx4,/m/06__v', '/m/05r655,/m/080hkjn', '/m/04yx4,/m/080hkjn',
    '/m/04yx4,/m/0584n8', '/m/01bl7v,/m/01940j', '/m/05r655,/m/01940j',
    '/m/03bt1vf,/m/01940j', '/m/03bt1vf,/m/071p9',
    '/m/01bl7v,/m/071p9', '/m/05r655,/m/071p9', '/m/05r655,/m/01s55n',
    '/m/01bl7v,/m/01s55n'])

def get_cover_iou(row):
    #assert row['XMin1'] <= row['XMax1']
    #assert row['YMin1'] <= row['YMax1']
    #assert row['XMin2'] <= row['XMax2']
    #assert row['YMin2'] <= row['YMax2']

    # determine the coordinates of the intersection rectangle
    x_left = max(row['XMin1'], row['XMin2'])
    y_top = max(row['YMin1'], row['YMin2'])
    x_right = min(row['XMax1'], row['XMax2'])
    y_bottom = min(row['YMax1'], row['YMax2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (row['XMax1'] - row['XMin1']) * (row['YMax1'] - row['YMin1'])
    bb2_area = (row['XMax2'] - row['XMin2']) * (row['YMax2'] - row['YMin2'])
    min_area = min(bb1_area, bb2_area)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    #iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-6)
    iou = intersection_area / (min_area + 1e-6)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_neg_sample(group):
    img_id, group = group
    n = len(group.LabelName.values)
    if n < 2:
        return []
    result = []
    max_sample_per_img = 100

    rows_c1 = [group.iloc[i] for i in range(group.shape[0]) if group.iloc[i].LabelName in classes_1]
    rows_c2 = [group.iloc[i] for i in range(group.shape[0]) if group.iloc[i].LabelName in classes_2]

    if len(rows_c1) < 1 or len(rows_c2) < 1:
        return []

    for row1 in rows_c1:
        for row2 in rows_c2:
            if len(result) >= max_sample_per_img:
                return result
            if ','.join([row1.LabelName, row2.LabelName]) in classes:
                result.append({
                    'ImageID': img_id,
                    'LabelName1': row1.LabelName,
                    'LabelName2': row2.LabelName,
                    'XMin1': row1.XMin,
                    'XMax1': row1.XMax,
                    'YMin1': row1.YMin,
                    'YMax1': row1.YMax,
                    'XMin2': row2.XMin,
                    'XMax2': row2.XMax,
                    'YMin2': row2.YMin,
                    'YMax2': row2.YMax,
                    'RelationshipLabel': 'none'
                })
    #print(len(result))
    return result

def create_train_data(args):
    df_box = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd-bbox.csv'))
    df_vrd = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd.csv'))
    df_pos = df_vrd.loc[df_vrd.RelationshipLabel=='wears'].copy()
    #df_pos.RelationshipLabel.value_counts()
    #classes_1 = df_pos.LabelName1.unique()
    #classes_2 = df_pos.LabelName2.unique()
    print('num box images:', len(df_box.ImageID.unique()))
    print('num wears images:', len(df_vrd.loc[df_vrd.RelationshipLabel=='wears'].ImageID.unique()))
    print('num pos samples:', len(df_pos))
    print('num vrd images:', len(df_vrd.ImageID.unique()))

    pos_img_ids = set(df_pos.ImageID.values)
    df_box_neg = df_box.loc[~df_box.ImageID.isin(pos_img_ids)]
    #df_box_neg['target'] = df_box_neg.LabelName1.str.cat(df_box_neg.LabelName2, sep=',')
    print('neg samples:', len(df_box_neg))
    df_box_neg = df_box_neg.loc[df_box_neg.LabelName.isin(classes_1 | classes_2)].copy()
    print('filtered neg samples:', len(df_box_neg))
    print('num pos images:', len(pos_img_ids))
    print('num neg images:', len(df_box_neg.ImageID.unique()))

    print('grouping negative samples')
    groups = list(df_box_neg.groupby('ImageID'))

    print('creating negative samples')
    with Pool(50) as p:
        samples = list(tqdm(iterable=p.imap_unordered(get_neg_sample, groups), total=len(groups)))

    neg_samples = []
    for results in tqdm(samples, total=len(samples)):
        for r in results:
            neg_samples.append(r)
    
    df_neg = pd.DataFrame(neg_samples)
    print('num of negative samples:', len(df_neg))
    print('saving to ', args.generate)
    df_neg.to_csv(args.generate, index=False)
    print('done')

def parallel_apply(df, func, n_cores=24):
    #ncores = 24
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def add_features(df):
    df['iou'] = df.apply(lambda row: get_iou(row), axis=1) 
    df['coveriou'] = df.apply(lambda row: get_cover_iou(row), axis=1) 
    df['size1'] = df.apply(lambda row: (row.XMax1 - row.XMin1) * (row.YMax1 - row.YMin1), axis=1)
    df['size2'] = df.apply(lambda row: (row.XMax2 - row.XMin2) * (row.YMax2 - row.YMin2), axis=1)
    df['xcenter1'] = df.apply(lambda row: (row.XMax1 + row.XMin1) / 2, axis=1)
    df['xcenter2'] = df.apply(lambda row: (row.XMax2 + row.XMin2) / 2, axis=1)
    df['ycenter1'] = df.apply(lambda row: (row.YMax1 + row.YMin1) / 2, axis=1)
    df['ycenter2'] = df.apply(lambda row: (row.YMax2 + row.YMin2) / 2, axis=1)
    df['aspect1'] = df.apply(lambda row: (row.XMax1 - row.XMin1) / (row.YMax1 - row.YMin1 + 1e-6), axis=1)
    df['aspect2'] = df.apply(lambda row: (row.XMax2 - row.XMin2) / (row.YMax2 - row.YMin2 + 1e-6), axis=1)
    df['xcenterdiff'] = df.apply(lambda row: ((row.XMax1 + row.XMin1) - (row.XMax2 + row.XMin2)) / 2, axis=1)
    df['ycenterdiff'] = df.apply(lambda row: ((row.YMax1 + row.YMin1) - (row.YMax2 + row.YMin2)) / 2, axis=1)
    return df

#img_files = glob.glob(settings.TRAIN_IMG_DIR + '/**/*.jpg')
#fullpath_dict = {}
#for fn in tqdm(img_files):
#    fullpath_dict[os.path.basename(fn).split('.')[0]] = fn

#def get_img_ratio(row):
#    w, h = get_image_size(fullpath_dict[row.ImageID])
#    return w / h

#def add_features(df):
#    df['iou'] = df.apply(lambda row: get_iou(row), axis=1)
#    df['ratio'] = df.apply(lambda row: get_img_ratio(row), axis=1)
#    return df

def get_train_data(args):
    df_vrd = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd.csv'))
    df_pos = df_vrd.loc[df_vrd.RelationshipLabel=='wears'].copy()
    #df_pos = df_vrd.loc[df_vrd.RelationshipLabel!='is'].copy()
    df_pos.RelationshipLabel = 1
    print(df_pos.head())

    df_neg = shuffle(pd.read_csv(args.neg_sample_fn)) #.iloc[:2500].copy()
    df_neg.RelationshipLabel = 0
    #df_neg.iloc[0].RelationshipLabel = 'xxx'
    print(df_neg.head())

    df_train = pd.concat([df_pos, df_neg], axis=0, sort=False)
    print(df_train.shape)

    df_train = shuffle(df_train)
    df_train = parallel_apply(df_train, add_features)
    print(df_train.head())
    print(df_train.dtypes)

    y = df_train.RelationshipLabel
    #print(y[:20])
    X = df_train.drop(['ImageID', 'RelationshipLabel'], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    return X_train, X_val, y_train, y_val

def train(args):
    X_train, X_val, y_train, y_val = get_train_data(args)
    print(X_train.dtypes)
    categorical_feature_indices = np.where(X_train.dtypes != np.float)[0]
    print(categorical_feature_indices)

    model = CatBoostClassifier(
        custom_loss = ['Accuracy'],
        #custom_metric = ['Accuracy'],
        #eval_metric = ['Accuracy'],

        iterations=1000, #2000,
        learning_rate=0.05,
        border_count=254,
        metric_period=10,
        #depth=5,
        task_type="GPU",
        verbose=True
    )
    #print(dir(model))

    #from_model = CatBoostClassifier()
    #from_model.load_model(args.model_file)

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_feature_indices,
        eval_set=(X_val, y_val),
        use_best_model=True,
        #plot=True #,
        #init_model=from_model
    )

    model.save_model(args.model_file)

def test_model():
    model = CatBoostClassifier() #task_type="GPU")
    model.load_model('insideof/cat_470_167.model')

    df_vrd = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd.csv'))
    df_pos = df_vrd.loc[df_vrd.RelationshipLabel=='under'].copy()
    #df_pos = pd.read_csv('insideof/df_neg.csv').iloc[3000:]
    df_pos = parallel_apply(df_pos, add_features)
    X = df_pos.drop(['ImageID', 'RelationshipLabel'], axis=1)
    p = model.predict_proba(X)
    y = model.predict(X)
    print(p[:100])
    print(y[:100] == 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='catboost model')
    parser.add_argument('--generate', type=str, default='wears/df_neg.csv')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--neg_sample_fn', type=str, default=None)
    parser.add_argument('--model_file', type=str, default='wears/cat.model')

    args = parser.parse_args()
    print(args)

    #test_model()
    #exit(0)

    if args.train:
        assert args.neg_sample_fn is not None
        train(args)
    elif args.generate:
        create_train_data(args)
