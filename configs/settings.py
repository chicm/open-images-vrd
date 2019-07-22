import os

#ROOT_DIR = r'/media/chicm/NVME/open-images'
ROOT_DIR = r'/mnt/chicm/data/open-images'

DETECT_DATA_DIR = os.path.join(ROOT_DIR, 'detect')

#TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'train', 'imgs', 'train_1-5')
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'train', 'imgs')

VAL_IMG_DIR = os.path.join(ROOT_DIR, 'val', 'imgs')
TEST_IMG_DIR = os.path.join(ROOT_DIR, 'test')
