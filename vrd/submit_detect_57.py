import os
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import tqdm
from vrd.utils import get_image_size, get_top_classes
import  settings

def get_preds(raw_pred, threshold=0):
    res = {
        'labels': [],
        'scores': [],
        'bboxes': []
    }
    for i, p in enumerate(raw_pred):
        if len(p) > 0:
            for e in p:
                if e[4] > threshold:
                    res['labels'].append(i)
                    res['scores'].append(e[4])
                    res['bboxes'].append(e[:4])
    res['labels'] = np.array(res['labels'])
    res['scores'] = np.array(res['scores'])
    res['bboxes'] = np.array(res['bboxes'])
    return res

def get_pred_str(pred, w, h, classes):
    res = []
    for label, score, bbox in zip(pred['labels'], pred['scores'], pred['bboxes']):
        res.append(classes[label])
        res.append('{:.7f}'.format(score))
        res.append('{:.7f}'.format(bbox[0]/w))
        res.append('{:.7f}'.format(bbox[1]/h))
        res.append('{:.7f}'.format(bbox[2]/w))
        res.append('{:.7f}'.format(bbox[3]/h))
    res = [str(x) for x in res]
    return ' '.join(res)

def create_submit(args, img_list_file='VRD_sample_submission.csv', img_dir=settings.TEST_IMG_DIR):
    with open(args.pred_file, 'rb') as f:
        print('loading: ', args.pred_file)
        preds = pickle.load(f)

    df_test = pd.read_csv(os.path.join(settings.DATA_DIR, img_list_file))
    print('getting image sizes')
    df_test['h'] = df_test.ImageId.map(lambda x: get_image_size(os.path.join(img_dir, '{}.jpg'.format(x)))[1])
    df_test['w'] = df_test.ImageId.map(lambda x: get_image_size(os.path.join(img_dir, '{}.jpg'.format(x)))[0])
    print(df_test.head())

    final_preds = []

    for p in tqdm(preds, total=len(preds)):
        final_preds.append(get_preds(p, args.th))
    total_objs = 0
    for p in final_preds:
        total_objs += len(p['labels'])
    print('total predicted objects:', total_objs)

    classes, _ = get_top_classes()
    pred_strs = []
    for i, p in tqdm(enumerate(final_preds), total=len(final_preds)):
        h = df_test.iloc[i].h
        w = df_test.iloc[i].w
        pred_strs.append(get_pred_str(p, w, h, classes))
    df_test['PredictionString'] = pred_strs
    print(df_test.head())
    df_test.to_csv(args.out, index=False, columns=['ImageId', 'PredictionString'])
    print('done')

def create_val_submission(args):
    create_submit(args, img_list_file='val_imgs.csv', img_dir=settings.VAL_IMG_DIR)

def create_val_prediction(args):
    #TODO
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission from pred file')
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--val_sub', action='store_true')
    parser.add_argument('--val_preds', action='store_true')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--th', type=float, default=0.)
    args = parser.parse_args()

    if args.val_sub:
        create_val_submission(args)
    elif args.val_preds:
        create_val_prediction(args)
    else:
        create_submit(args)
