import os
import argparse
import pandas as pd
import numpy as np
import time
import math
from multiprocessing import Pool
from catboost import CatBoostClassifier
from utils import get_iou
from train_catboost import classes_1, classes_2

model = None

def get_sort_score(row):
    return math.sqrt(row.confidence1 * row.confidence2) * row.coef

def get_pred_str(row):
    conf = round(row.sort_score,6)
    rel = [conf, row.LabelName1, row.XMin1, row.YMin1, row.XMax1, row.YMax1,
               row.LabelName2, row.XMin2, row.YMin2, row.XMax2, row.YMax2, row.RelationshipLabel]
    rel = [str(x) for x in rel]
    return ' '.join(rel)

def fast_get_prediction_string(test_row):
    dets = test_row.dets
    cur_test_dets = []
    #dets = sorted(dets, key=lambda x: x[1], reverse=True)  #[:35]
    for i in range(len(dets)):
        for j in range(len(dets)):
            det1, det2 = dets[i], dets[j]
            if i != j and det1[0] in set(classes_1) and det2[0] in set(classes_2):
                cur_test_dets.append({
                    'ImageID': test_row.ImageId,
                    'LabelName1': det1[0],
                    'LabelName2': det2[0],
                    'confidence1': float(det1[1]),
                    'confidence2': float(det2[1]),
                    'XMin1': float(det1[2]),
                    'YMin1': float(det1[3]),
                    'XMax1': float(det1[4]),
                    'YMax1': float(det1[5]),
                    'XMin2': float(det2[2]),
                    'YMin2': float(det2[3]),
                    'XMax2': float(det2[4]),
                    'YMax2': float(det2[5])
                })
    
    if len(cur_test_dets) < 1:
        return ''

    cur_df = pd.DataFrame(cur_test_dets)
    cur_df['iou'] = cur_df.apply(lambda row: get_iou(row), axis=1)
    cur_df = cur_df[['ImageID', 'LabelName1', 'LabelName2', 'XMin1', 
                   'XMax1', 'YMin1', 'YMax1', 'XMin2', 'XMax2', 'YMin2', 'YMax2', 'iou', 'confidence1', 'confidence2']]
    #print(cur_df.head())
    cur_x_test = cur_df.drop(['ImageID', 'confidence1', 'confidence2'], axis=1)
    pred_score = model.predict_proba(cur_x_test)
    pred_rel = model.predict(cur_x_test)
    
    
    cur_x_test['coef'] = 1 - pred_score[:, 5]
    cur_x_test['confidence1'] = cur_df['confidence1']
    cur_x_test['confidence2'] = cur_df['confidence2']
    cur_x_test['RelationshipLabel'] = pred_rel
    cur_x_test['sort_score'] = cur_x_test.apply(lambda row: get_sort_score(row), axis=1)
    
    cur_x_test = cur_x_test.loc[cur_x_test.RelationshipLabel != 'none'].copy()

    cur_x_test = cur_x_test.nlargest(200, 'sort_score').copy()
    #cur_x_test.sort_values(by='sort_score', axis=0, ascending=False, inplace=False, kind='quicksort')
    #cur_x_test = cur_x_test[:200].copy()

    if len(cur_x_test) < 1:
        return ''

    cur_x_test['PredictionString'] = cur_x_test.apply(lambda row: get_pred_str(row), axis=1)
    #print(cur_x_test.head())
    
    return ' '.join(cur_x_test.PredictionString.values)

tuple_classes = set(classes_1) | set(classes_2)

def get_det(pred_str):
    dets = []
    det = []
    for i, e in enumerate(pred_str.split(' ')):
        if i % 6 == 0:
            det = []
        det.append(e)
        if (i+1) % 6 == 0:
            if det[0] in set(tuple_classes): #and float(det[1]) > 0.1:
                dets.append(det)
            
    return dets

def add_pred_string(df):
    df['PredictionString'] = df.apply(lambda row: fast_get_prediction_string(row), axis=1)
    return df

def parallel_apply(df, func, n_cores=24):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def submit(args):
    global model
    model = CatBoostClassifier() #task_type="GPU")
    print('loading {}...'.format(args.model_file))
    model.load_model(args.model_file)
    #print(dir(model))
    print(model.classes_)
    #print(model.feature_importances_)
    print(model._tree_count)
    print(model.learning_rate_)

    #exit(0)

    print('loading {}...'.format(args.detect_pred_file))
    df_det = pd.read_csv(args.detect_pred_file)
    df_det['dets'] = df_det.PredictionString.map(lambda x: get_det(str(x)))
    print('detected objs:', df_det.dets.map(lambda x: len(x)).sum())

    print('predicting...')
    df_sub = df_det.copy()
    df_sub.PredictionString = ''

    bg = time.time()
    df_sub = parallel_apply(df_sub, add_pred_string)
    df_sub.to_csv(args.out, columns=['ImageId', 'PredictionString'], index=False)

    print('Done, total time:', time.time() - bg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission from detect pred file')
    parser.add_argument('--detect_pred_file', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    submit(args)
