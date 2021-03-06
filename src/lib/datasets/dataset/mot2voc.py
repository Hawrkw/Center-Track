import os
import cv2
import os
import numpy as np
import time
import argparse
import shutil
import codecs
import progressbar
import sys
#sys.path.append('/home/ivo/PycharmProjects/CenterNet-Track/')
train_17 = ['/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-02/',
            '/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-04/',
            '/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-05/',
            '/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-11/',
            '/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-13/']

val_17 = ['/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/val/MOT17-09/',
           '/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/val/MOT17-10/',]

train_20 = ['data/MOT20Det/train/MOT20-01/',
            'data/MOT20Det/train/MOT20-02/',
            'data/MOT20Det/train/MOT20-03/',
            'data/MOT20Det/train/MOT20-05/']

val_20 = ['MOT20Det/test/MOT20-04/',
           'MOT20Det/test/MOT20-06/',
           'MOT20Det/test/MOT20-07/',
           'MOT20Det/test/MOT20-08/']


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MOT2VOC format')
    parser.add_argument('--year',choices=['17', '20'],default='17',type=str,help='year of MOT dataset')
    args = parser.parse_args()
    return args


def parse_ini(dir):
    ini_fp = open(dir + 'seqinfo.ini', 'r')
    seq_info = ini_fp.readlines()
    seqLenth = int(seq_info[4][10:])
    imWidth = int(seq_info[5][8:])
    imHeight = int(seq_info[6][9:])
    return seqLenth, imWidth, imHeight


def gennerate_gt(gt, Annotation, frame, filename, width, height):
    fp_gt = open(gt)
    gt_lines = fp_gt.readlines()

    gt_fram = []
    for line in gt_lines:
        fram_id = int(line.split(',')[0])
        if fram_id == frame:
            visible = float(line.split(',')[8])
            label_class = line.split(',')[7]
            if (label_class == '1' or label_class == '2' or label_class == '7') and visible > 0.3:
                gt_fram.append(line)

    with codecs.open(Annotation + filename + '.xml', 'w') as xml:
        xml.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'voc' + '</folder>\n')
        xml.write('\t<filename>' + filename + '.jpg' + '</filename>\n')
        # xml.write('\t<path>' + path + "/" + info1 + '</path>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database> The MOT-Det </database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>' + str(width) + '</width>\n')
        xml.write('\t\t<height>' + str(height) + '</height>\n')
        xml.write('\t\t<depth>' + '3' + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for bbox in gt_fram:
            x1 = int(bbox.split(',')[2])
            y1 = int(bbox.split(',')[3])
            x2 = int(bbox.split(',')[4])
            y2 = int(bbox.split(',')[5])

            xml.write('\t<object>\n')
            xml.write('\t\t<name>person</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>0</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(x1) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(y1) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(x1 + x2) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(y1 + y2) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
        xml.write('</annotation>')


# 用于校验图片数量和标注数量是否一致
def check_num(data_dir, JPEGImage_dir, Annotations_dir=None, ori_num=0):
    num = 0
    for folder in data_dir:
        folder_len, _, _ = parse_ini(folder)
        num += folder_len
    img_list = os.listdir(JPEGImage_dir)
    if ori_num == 0:
        img_num = len(img_list)
    else:
        img_num = len(img_list) - ori_num
    # print('img_num:',img_num)
    if Annotations_dir:
        ann_list = os.listdir(Annotations_dir)
        ann_num = len(ann_list)
        assert ann_num == num
    assert img_num == num, 'if it is the second time run this demo, please delete the JPEGImages folder and retry'
    # print('num:', num)
    print('folders {} have been succeed checked'.format(data_dir))
    return num


def segment_dataset(ImageSets, Main, thr1=0.8, thr2=0.9):
    fp_all = open(ImageSets + 'train_all.txt', 'r')
    fp_train = open(Main + 'train.txt', 'w')
    fp_test = open(Main + 'test.txt', 'w')
    fp_val = open(Main + 'val.txt', 'w')
    train_list = fp_all.readlines()
    print(len(train_list))

    for line in train_list:
        rand_a = np.random.rand(1)
        if rand_a <= thr1:
            fp_train.writelines(line)
        if rand_a > thr1:
            fp_val.writelines(line)
        # if rand_a > thr1 and rand_a <= thr2:
        #     fp_val.writelines(line)
        # if rand_a > thr2 and rand_a <= 1:
        #     fp_test.writelines(line)
    fp_train.close()
    fp_val.close()
    #fp_test.close()

    print('segment the MOT dataset into train,val,test subsets')


def transform_train():
    args = parse_args()
    if args.year == '17':
        train_dirs = train_17
        val_dirs = val_17
    if args.year == '20':
        train_dirs = train_20
        val_dirs = val_20

    motyear = args.year
    folder = '/home/ivo/PycharmProjects/CenterNet-Track/data/' + 'MOT' + motyear + 'Det' + '/train/'
    Annotations = folder + 'Annotations/'
    ImageSets = folder + 'ImageSets/'
    JPEGImages = folder + 'JPEGImages/'
    Main = ImageSets + 'Main/'
    if not os.path.exists(Annotations):
        os.makedirs(Annotations)
    if not os.path.exists(ImageSets):
        os.makedirs(ImageSets)
    if not os.path.exists(JPEGImages):
        os.makedirs(JPEGImages)
    if not os.path.exists(Main):
        os.makedirs(Main)

    fp_txt = open(ImageSets + 'train_all.txt', 'w')

    for train_ in train_dirs:
        seqLenth, imWidth, imHeight = parse_ini(train_)
        img1 = train_ + 'img1/'
        gt = train_ + 'gt/gt.txt'
        folder_id = train_[-3:-1]
        img_list = os.listdir(img1)

        assert len(img_list) == seqLenth
        print('start')
        for img in img_list:

            format_name = args.year + folder_id + img
            fp_txt.writelines(format_name[:-4] + '\n')  # 将生成的新的文件名写入train_all.txt，用于后续数据集拆分
            shutil.copy(img1 + img, JPEGImages + '/' + format_name)  # 将文件移动到指定文件夹并重新命名
            frame = int(img[:-4])
            gennerate_gt(gt, Annotation=Annotations, frame=frame, filename=format_name[:-4], width=imWidth,
                         height=imHeight)  # 生成标注文件

    fp_txt.close()

    train_num = check_num(train_dirs, JPEGImages, Annotations)
    #segment_dataset(ImageSets, Main)

def transform_val():
    args = parse_args()
    if args.year == '17':
        train_dirs = train_17
        val_dirs = val_17
    if args.year == '20':
        train_dirs = train_20
        val_dirs = val_20

    motyear = args.year
    folder = 'data/' + 'MOT' + motyear + 'Det' + '/val/'
    Annotations = folder + 'Annotations/'
    ImageSets = folder + 'ImageSets/'
    JPEGImages = folder + 'JPEGImages/'
    Main = ImageSets + 'Main/'
    if not os.path.exists(Annotations):
        os.makedirs(Annotations)
    if not os.path.exists(ImageSets):
        os.makedirs(ImageSets)
    if not os.path.exists(JPEGImages):
        os.makedirs(JPEGImages)
    if not os.path.exists(Main):
        os.makedirs(Main)

    fp_txt = open(ImageSets + 'val_all.txt', 'w')

    for val_ in val_dirs:
        seqLenth, imWidth, imHeight = parse_ini(val_)
        img1 = val_ + 'img1/'
        gt = val_ + 'gt/gt.txt'
        folder_id = val_[-3:-1]
        img_list = os.listdir(img1)

        assert len(img_list) == seqLenth
        print('start')
        for img in img_list:
            format_name = args.year + folder_id + img
            fp_txt.writelines(format_name[:-4] + '\n')  # 将生成的新的文件名写入train_all.txt，用于后续数据集拆分
            shutil.copy(img1 + img, JPEGImages + '/' + format_name)  # 将文件移动到指定文件夹并重新命名
            frame = int(img[:-4])
            gennerate_gt(gt, Annotation=Annotations, frame=frame, filename=format_name[:-4], width=imWidth,
                         height=imHeight)  # 生成标注文件

    fp_txt.close()

    val_num = check_num(val_dirs, JPEGImages, Annotations)
    print(val_num)

if __name__ == '__main__':
    transform_train()
    #transform_val()