import os
import argparse
import pandas as pd
import numpy as np
import time
import math
from multiprocessing import Pool
from catboost import CatBoostClassifier
from utils import get_iou
from train_inside_of import classes_1, classes_2, classes, add_features, parallel_apply

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
            if i != j and ','.join([det1[0], det2[0]]) in classes:
                cur_test_dets.append({
                    #'ImageID': test_row.ImageId,
                    'LabelName1': det1[0],
                    'LabelName2': det2[0],
                    'XMin1': float(det1[2]),
                    'XMax1': float(det1[4]),
                    'YMin1': float(det1[3]),
                    'YMax1': float(det1[5]),
                    'XMin2': float(det2[2]),
                    'XMax2': float(det2[4]),
                    'YMin2': float(det2[3]),
                    'YMax2': float(det2[5]),
                    'confidence1': float(det1[1]),
                    'confidence2': float(det2[1]),
                })
    
    if len(cur_test_dets) < 1:
        return ''

    cur_df = pd.DataFrame(cur_test_dets)

    cur_df = add_features(cur_df)
    cur_x_test = cur_df.drop(['confidence1', 'confidence2'], axis=1)
    
    pred_score = model.predict_proba(cur_x_test)
    pred_rel = model.predict(cur_x_test)

    #cur_x_test['coef'] = 1 - pred_score[:, 5]
    cur_x_test['coef'] = pred_score[:, 1]
    cur_x_test['confidence1'] = cur_df['confidence1']
    cur_x_test['confidence2'] = cur_df['confidence2']
    cur_x_test['RelationshipLabel'] = pd.Series(pred_rel).map(lambda x: 'inside_of' if x == 1 else 'none')
    cur_x_test['sort_score'] = cur_x_test.apply(lambda row: get_sort_score(row), axis=1)
    
    cur_x_test = cur_x_test.loc[cur_x_test.RelationshipLabel != 'none'].copy()

    cur_x_test = cur_x_test.nlargest(20, 'sort_score').copy()
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

class CatBoostEnsembleModel:
    def __init__(self):
        self.model1 = CatBoostClassifier()
        self.model2 = CatBoostClassifier()
        self.model1.load_model('lb23578/cat_154k_144_1000.model')
        self.model2.load_model('lb22592/cat_0820_500_143.model')
        print(self.model1.classes_)

    def predict_with_proba(self, X, w=[0.7, 0.3]):
        p1 = self.model1.predict_proba(X)
        p2 = self.model2.predict_proba(X)
        prob = p1*w[0] + p2*w[1]
        idx = np.argmax(prob, axis=1)
        assert len(idx) == len(prob)
        labels = np.array(self.model1.classes_)[idx]
        assert len(labels) == len(prob)

        return labels, prob

def submit(args):
    global model
    #model = CatBoostEnsembleModel()
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


def submit_val_preds(args):
    def get_pred_dets(pred_str):
        if len(pred_str) < 1:
            return []
        dets = []
        det = []
        for i, e in enumerate(pred_str.split(' ')):
            if i % 12 == 0:
                det = []
            det.append(e)
            if (i+1) % 12 == 0:
                dets.append(det)
                
        return dets

    submit(args)

    df_sub = pd.read_csv(args.out)
    df_sub.PredictionString = df_sub.PredictionString.fillna('')
    res = []
    for i, row in df_sub.iterrows():
        objs = get_pred_dets(row.PredictionString)
        for o in objs:
            det_obj = {
                'ImageID': row.ImageId,
                'LabelName1': o[1],
                'XMin1': o[2],
                'YMin1': o[3],
                'XMax1': o[4],
                'YMax1': o[5],
                'LabelName2': o[6],
                'XMin2': o[7],
                'YMin2': o[8],
                'XMax2': o[9],
                'YMax2': o[10],
                'RelationshipLabel': o[11],
                'Score': o[0]
            }
            res.append(det_obj)

    df_val_sub = pd.DataFrame(res)
    df_val_sub.to_csv(args.out, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission from detect pred file')
    parser.add_argument('--detect_pred_file', type=str, required=True)
    parser.add_argument('--model_file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--val', action='store_true')
    args = parser.parse_args()

    if args.val:
        submit_val_preds(args)
    else:
        submit(args)
