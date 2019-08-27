import os.path as osp
import glob
import pandas as pd

DATA_DIR = '/mnt/chicm/data/open-images/relation'
IMG_DIR = '/mnt/chicm/data/open-images/val'

def create_val_img_meta():
    filenames = glob.glob(IMG_DIR+'/*.jpg')
    filenames = [osp.basename(x).split('.')[0] for x in filenames]
    print(filenames[:2])
    filenames = set(filenames)

    df_vrd = pd.read_csv(osp.join(DATA_DIR, 'challenge-2019-validation-vrd.csv'))
    print(df_vrd.shape)
    img_ids = df_vrd.ImageID.unique()
    meta_list = []
    for img_id in img_ids:
        meta_list.append({'ImageId': img_id})
        assert img_id in filenames
    df_val_img = pd.DataFrame(meta_list)
    print(df_val_img.head())
    print(df_val_img.shape)
    df_val_img.to_csv(osp.join(DATA_DIR, 'val_imgs.csv'), index=False)


#import argparse
if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description='create mmdetection dataset')
    #parser.add_argument('--img_dirs', type=str, default=None)
    #parser.add_argument('--test_img_dir', type=str, default=None)
    #parser.add_argument('--output', type=str, required=True)
    #parser.add_argument('--test', action='store_true')
    #parser.add_argument('--flat', action='store_true')
    #parser.add_argument('--meta', type=str, default='challenge-2019-train-detection-bbox.csv')
    #parser.add_argument('--top_classes', action='store_true')
    #parser.add_argument('--start_index', type=int)
    #parser.add_argument('--end_index', type=int)
    #args = parser.parse_args()

    create_val_img_meta()
