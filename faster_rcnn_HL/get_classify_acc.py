import glob
import json
import math
import operator
import os
import shutil
import sys
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass
import cv2
import numpy as np

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def get_acc():
    path = './map_out'
    GT_PATH = os.path.join(path, 'ground-truth')
    DR_PATH = os.path.join(path, 'detection-results')
    
    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    # detection_result_file_list = glob.glob(DR_PATH + '/*.txt')
    ground_truth_files_list.sort()
    # detection_result_file_list.sort()
    
    # gt_counter_per_class = {}
    # counter_images_per_class = {}
    total = 0
    correct = 0
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        lines_list_DR = file_lines_to_list(temp_path)
        lines_list_GT = file_lines_to_list(txt_file)
        GT_bounding_boxes = []
        DR_bounding_boxes = []
        already_seen_classes = []
        for line in lines_list_GT:
            if "difficult" in line:
                class_name, left, top, right, bottom, _difficult = line.split()
            else:
                class_name, left, top, right, bottom = line.split()
            # get gtloc
            bbox = left + " " + top + " " + right + " " + bottom
            GT_bounding_boxes.append({"class_name": class_name, "bbox": bbox})
        for line in lines_list_DR:
            class_name, _, left, top, right, bottom = line.split()
            # get gtloc
            bbox = left + " " + top + " " + right + " " + bottom
            DR_bounding_boxes.append({"class_name": class_name, "bbox": bbox})
    
        for i in range(len(GT_bounding_boxes)):
            gt_box = GT_bounding_boxes[i]
            bbgt = [float(x) for x in gt_box["bbox"].split()]
            max_iou = -1
            max_label = ''
            for j in range(len(DR_bounding_boxes)):
                dt_box = DR_bounding_boxes[j]
                bb = [float(x) for x in dt_box["bbox"].split()]
                bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > max_iou:
                        max_iou = ov
                        max_label = dt_box["class_name"]
            if max_iou >= 0.5:
                total += 1
                if gt_box["class_name"] == max_label:
                    correct += 1
    # print(total)
    # print(correct)
    acc = correct/total
    print("acc:", acc)
    return acc


