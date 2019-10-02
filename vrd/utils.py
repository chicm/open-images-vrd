import os
import glob
import pandas as pd
import struct
import imghdr
import cv2

import settings

def get_train_img_fullpath_dict():
    img_files = glob.glob(settings.TRAIN_IMG_DIR + '/**/*.jpg')
    print('num image files:', len(img_files))
    fullpath_dict = {}
    for fn in img_files:
        fullpath_dict[os.path.basename(fn).split('.')[0]] = fn
    return fullpath_dict

def get_top_classes(start_index=0, end_index=57):
    df = pd.read_csv(os.path.join(settings.DATA_DIR, 'top_classes.csv'))
    #print(df.shape)
    c = df['class'].values[start_index:end_index]
    #print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            raise AssertionError('imghead len != 24')
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                raise AssertionError('png check failed')
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                raise
        else:
            print(fname, imghdr.what(fname))
            #raise AssertionError('file format not supported')
            img = cv2.imread(fname)
            print(img.shape)
            height, width, _ = img.shape

        return width, height

def get_iou(row):
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

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + 1e-6)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def add_XYinter(sub_final_nonis):
    tmp1 = pd.concat([sub_final_nonis['XMax1'] - sub_final_nonis['XMin2'],\
               sub_final_nonis['XMax2'] - sub_final_nonis['XMin1']],axis=1).max(axis=1)
    tmp2 = sub_final_nonis['XMax1'] - sub_final_nonis['XMin1'] + \
               sub_final_nonis['XMax2'] - sub_final_nonis['XMin2']
    tmp3 = tmp1/tmp2
    tmp4 = pd.concat([sub_final_nonis['YMax1'] - sub_final_nonis['YMin2'],\
               sub_final_nonis['YMax2'] - sub_final_nonis['YMin1']],axis=1).max(axis=1)
    tmp5 = sub_final_nonis['YMax1'] - sub_final_nonis['YMin1'] + \
               sub_final_nonis['YMax2'] - sub_final_nonis['YMin2']
    tmp6 = tmp4/tmp5
    sub_final_nonis['Xinter'] = tmp3
    sub_final_nonis['Yinter'] = tmp6
    return sub_final_nonis


if __name__ == '__main__':
    get_top_classes()