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

classes_1 = set(['/m/04yx4', '/m/01bl7v', '/m/05r655', '/m/03bt1vf'])
classes_2 = set(['/m/01226z', '/m/0wdt60w', '/m/05ctyq'])

classes = set([
    '/m/04yx4,/m/01226z', '/m/04yx4,/m/0wdt60w', '/m/01bl7v,/m/01226z',
    '/m/05r655,/m/01226z', '/m/03bt1vf,/m/01226z',
    '/m/04yx4,/m/05ctyq', '/m/03bt1vf,/m/05ctyq',
    '/m/05r655,/m/05ctyq'])

def get_neg_sample(group):
    img_id, group = group
    n = len(group.LabelName.values)
    if n < 2:
        return []
    used = set()
    result = []
    for _ in range(50):
        idx1 = random.choice(list(range(n)))
        idx2 = random.choice(list(range(n)))
        if (idx1 != idx2) and ((idx1, idx2) not in used):
            row1 = group.iloc[idx1]
            row2 = group.iloc[idx2]
            label_name1 = row1.LabelName
            label_name2 = row2.LabelName
            #if label_name1 in set(classes_1) and label_name2 in set(classes_2):
            if ','.join([label_name1, label_name2]) in classes:
                result.append({
                    'ImageID': img_id,
                    'LabelName1': label_name1,
                    'LabelName2': label_name2,
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
                #result.append((group.iloc[idx1], group.iloc[idx2]))
                used.add((idx1, idx2))
        if len(used) >= 25:
            break
    #print(len(result))
    return result

def create_train_data(args):
    df_box = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd-bbox.csv'))
    df_vrd = pd.read_csv(os.path.join(DATA_DIR, 'challenge-2019-train-vrd.csv'))
    df_pos = df_vrd.loc[df_vrd.RelationshipLabel=='hits'].copy()
    #df_pos.RelationshipLabel.value_counts()
    #classes_1 = df_pos.LabelName1.unique()
    #classes_2 = df_pos.LabelName2.unique()
    print('num box images:', len(df_box.ImageID.unique()))
    print('num hits images:', len(df_vrd.loc[df_vrd.RelationshipLabel=='hits'].ImageID.unique()))
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
    df_pos = df_vrd.loc[df_vrd.RelationshipLabel=='hits'].copy()
    #df_pos = df_vrd.loc[df_vrd.RelationshipLabel!='is'].copy()
    df_pos.RelationshipLabel = 1
    print(df_pos.head())

    df_neg = shuffle(pd.read_csv(args.neg_sample_fn)).iloc[:500]
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
        learning_rate=0.1,
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='catboost model')
    parser.add_argument('--generate', type=str, default='hits/df_neg.csv')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--neg_sample_fn', type=str, default=None)
    parser.add_argument('--model_file', type=str, default='hits/cat.model')

    args = parser.parse_args()
    print(args)

    #test_model()
    #exit(0)

    if args.train:
        assert args.neg_sample_fn is not None
        train(args)
    elif args.generate:
        create_train_data(args)
