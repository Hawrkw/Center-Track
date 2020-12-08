import numpy as np
import json
import sys
import os
from collections import OrderedDict


def ltwh2ltrb(ltwh, width, height):
    ltrb = np.asarray([ltwh[0], ltwh[1], ltwh[0] + ltwh[2], ltwh[1] + ltwh[3]])
    ltrb[ltrb <= 0] = 0
    ltrb[0] = ltrb[0] if ltrb[0] <= width else width
    ltrb[1] = ltrb[1] if ltrb[1] <= height else height
    ltrb[2] = ltrb[2] if ltrb[2] <= width else width
    ltrb[3] = ltrb[3] if ltrb[3] <= height else height
    return ltrb.tolist()


def GtToJson(inputfile, width=768, height=576):
    """
    This function transform Gt file format to coco data format.
    Gt file is download in "https://motchallenge.net/".
    Usage:
    ex) $ python3 make.py gt/gt.txt
    """
    f = open(inputfile, 'r')
    lines = f.readlines()
    entire = []
    last_index = -1
    is_first = True
    folder_name = []
    folder_name = inputfile.split('/')[-3]
    folder_name = folder_name if folder_name else './'

    for i, line in enumerate(lines):
        line_list = line.split(',')
        line_list = [int(float(x)) for x in line_list]
        if (line_list[7] != 1 and line_list[7] != 2 and line_list[7] != 7) and line_list[8] < 0.3:
            continue
        new_index = line_list[0]
        ltwh = line_list[2:6]

        if last_index != new_index:
            if is_first:
                is_first = False
            else:
                group_data["ann"] = ann
                entire.append(group_data)

            group_data = OrderedDict()
            #file_name = MOT17-02/img1/000002.jpg---需要修改
            file_name = folder_name + "/img1/{0:06}.jpg".format(int(new_index))  # file format
            group_data["filename"] = file_name
            group_data["width"] = width
            group_data["height"] = height

            ann = OrderedDict()

            ann["bboxes"] = []
            ann["labels"] = []
            ann["bboxes_ignore"] = []
            ann["labels_ignore"] = []

            ann["bboxes"].append(ltwh2ltrb(ltwh, width, height))
            ann["labels"].append(7)
            last_index = new_index

        else:
            ann["bboxes"].append(ltwh2ltrb(ltwh, width, height))
            ann["labels"].append(7)
            last_index = new_index

        if i + 1 == len(lines):
            group_data["ann"] = ann
            entire.append(group_data)

    f.close()
    with open('json_from_{}.json'.format(inputfile.split('/')[-3]), 'w', encoding="utf-8") as make_file:
        json.dump(entire, make_file, ensure_ascii=False, indent="\t")
        print(f"\nfile_name : {make_file.name}\nwidth : {width}, height : {height}\n")
        os.chmod(make_file.name, 0o777)


if __name__ == '__main__':
    # if (len(sys.argv) < 2):
    #     print("please run 'make.py gt.txt [width, height]' format")
    #     sys.exit(1)
    # if len(sys.argv) == 4:
    #     GtToJson(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    # elif len(sys.argv) == 2:
    #     GtToJson(sys.argv[1])
    # else:
    #     print("please run 'make.py gt.txt [width, height]' format")
    #     sys.exit(1)
    GtToJson(inputfile='/home/ivo/PycharmProjects/CenterNet-Track/data/MOT17Det/train/MOT17-02/gt/gt.txt',
             width=1920,height=1080)
