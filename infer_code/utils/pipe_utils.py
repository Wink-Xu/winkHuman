# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import ast
import argparse
import glob
import yaml
import copy
import numpy as np

#from python.keypoint_preprocess import EvalAffine, TopDownEvalAffine, expand_crop




class Times(object):
    def __init__(self):
        self.time = 0.
        # start time
        self.st = 0.
        # end time
        self.et = 0.

    def start(self):
        self.st = time.time()

    def end(self, repeats=1, accumulative=True):
        self.et = time.time()
        if accumulative:
            self.time += (self.et - self.st) / repeats
        else:
            self.time = (self.et - self.st) / repeats

    def reset(self):
        self.time = 0.
        self.st = 0.
        self.et = 0.

    def value(self):
        return round(self.time, 4)


class PipeTimer(Times):
    def __init__(self):
        super(PipeTimer, self).__init__()
        self.total_time = Times()
        self.module_time = {
            'det': Times(),
            'mot': Times(),
            'attr': Times(),
            'kpt': Times(),
            'action': Times(),
            'reid': Times()
        }
        self.img_num = 0

    def get_total_time(self):
        total_time = self.total_time.value()
        total_time = round(total_time, 4)
        average_latency = total_time / max(1, self.img_num)
        qps = 0
        if total_time > 0:
            qps = 1 / average_latency
        return total_time, average_latency, qps

    def info(self):
        total_time, average_latency, qps = self.get_total_time()
        print("------------------ Inference Time Info ----------------------")
        print("total_time(ms): {}, img_num: {}".format(total_time * 1000,
                                                       self.img_num))

        for k, v in self.module_time.items():
            v_time = round(v.value(), 4)
            if v_time > 0:
                print("{} time(ms): {}".format(k, v_time * 1000))

        print("average latency time(ms): {:.2f}, QPS: {:2f}".format(
            average_latency * 1000, qps))
        return qps

    def report(self, average=False):
        dic = {}
        dic['total'] = round(self.total_time.value() / max(1, self.img_num),
                             4) if average else self.total_time.value()
        dic['det'] = round(self.module_time['det'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['det'].value()
        dic['mot'] = round(self.module_time['mot'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['mot'].value()
        dic['attr'] = round(self.module_time['attr'].value() /
                            max(1, self.img_num),
                            4) if average else self.module_time['attr'].value()
        dic['kpt'] = round(self.module_time['kpt'].value() /
                           max(1, self.img_num),
                           4) if average else self.module_time['kpt'].value()
        dic['action'] = round(
            self.module_time['action'].value() / max(1, self.img_num),
            4) if average else self.module_time['action'].value()

        dic['img_num'] = self.img_num
        return dic


def merge_model_dir(args, model_dir):
    # set --model_dir DET=ppyoloe/ to overwrite the model_dir in config file
    task_set = ['DET', 'ATTR', 'MOT', 'KPT', 'ACTION', 'REID']
    if not model_dir:
        return args
    for md in model_dir:
        md = md.strip()
        k, v = md.split('=', 1)
        k_upper = k.upper()
        assert k_upper in task_set, 'Illegal type of task, expect task are: {}, but received {}'.format(
            task_set, k)
        args[k_upper].update({'model_dir': v})
    return args


def merge_cfg(args):
    with open(args.config) as f:
        pred_config = yaml.safe_load(f)

    def merge(cfg, arg):
        merge_cfg = copy.deepcopy(cfg)
        for k, v in cfg.items():
            if k in arg:
                merge_cfg[k] = arg[k]
            else:
                if isinstance(v, dict):
                    merge_cfg[k] = merge(v, arg)
        return merge_cfg

    args_dict = vars(args)
    model_dir = args_dict.pop('model_dir')
    pred_config = merge_model_dir(pred_config, model_dir)
    pred_config = merge(pred_config, args_dict)
    return pred_config


def print_arguments(cfg):
    print('-----------  Running Arguments -----------')
    buffer = yaml.dump(cfg)
    print(buffer)
    print('------------------------------------------')


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    print("Found {} inference images in total.".format(len(images)))

    return images

def expand_crop(images, rect, expand_ratio=0.3):
    imgh, imgw, c = images.shape
    xmin, ymin, xmax, ymax, track_id, label, conf = [int(x) for x in rect]
    if label != 0:
        return None, None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    if h_half > w_half * 4 / 3:
        w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half))
    ymax = min(imgh - 1, int(center[0] + h_half))
    xmin = max(0, int(center[1] - w_half))
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect

def normal_crop(image, rect):
    imgh, imgw, c = image.shape
    xmin, ymin, xmax, ymax, track_id, label, conf = [int(x) for x in rect]
    org_rect = [xmin, ymin, xmax, ymax]
    if label != 0:
        return None, None, None
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(imgw, xmax)
    ymax = min(imgh, ymax)
    return image[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect


def crop_image_with_mot(input, mot_res, expand=True):
    res = mot_res['boxes']
    crop_res = []
    new_bboxes = []
    ori_bboxes = []
    for box in res:
        if expand:
            crop_image, new_bbox, ori_bbox = expand_crop(input, box)
        else:
            crop_image, new_bbox, ori_bbox = normal_crop(input, box)
        if crop_image is not None:
            crop_res.append(crop_image)
            new_bboxes.append(new_bbox)
            ori_bboxes.append(ori_bbox)
    return crop_res, new_bboxes, ori_bboxes

def translate_to_ori_images(keypoint_result, batch_records):
    kpts = keypoint_result[:, :, 0:2]
    scores = keypoint_result[:, :, 2]
    kpts[..., 0] += batch_records[:, 0:1]
    kpts[..., 1] += batch_records[:, 1:2]
    return kpts, scores


# def crop_image_with_det(batch_input, det_res, thresh=0.3):
#     boxes = det_res['boxes']
#     score = det_res['boxes'][:, 1]
#     boxes_num = det_res['boxes_num']
#     start_idx = 0
#     crop_res = []
#     for b_id, input in enumerate(batch_input):
#         boxes_num_i = boxes_num[b_id]
#         if boxes_num_i == 0:
#             continue
#         boxes_i = boxes[start_idx:start_idx + boxes_num_i, :]
#         score_i = score[start_idx:start_idx + boxes_num_i]
#         res = []
#         for box, s in zip(boxes_i, score_i):
#             if s > thresh:
#                 crop_image, new_box, ori_box = expand_crop(input, box)
#                 if crop_image is not None:
#                     res.append(crop_image)
#         crop_res.append(res)
#     return crop_res




# def parse_mot_res(input):
#     mot_res = []
#     boxes, scores, ids = input[0]
#     for box, score, i in zip(boxes[0], scores[0], ids[0]):
#         xmin, ymin, w, h = box
#         res = [i, 0, score, xmin, ymin, xmin + w, ymin + h]
#         mot_res.append(res)
#     return {'boxes': np.array(mot_res)}


# def refine_keypoint_coordinary(kpts, bbox, coord_size):
#     """
#         This function is used to adjust coordinate values to a fixed scale.
#     """
#     tl = bbox[:, 0:2]
#     wh = bbox[:, 2:] - tl
#     tl = np.expand_dims(np.transpose(tl, (1, 0)), (2, 3))
#     wh = np.expand_dims(np.transpose(wh, (1, 0)), (2, 3))
#     target_w, target_h = coord_size
#     res = (kpts - tl) / wh * np.expand_dims(
#         np.array([[target_w], [target_h]]), (2, 3))
#     return res


# def parse_mot_keypoint(input, coord_size):
#     parsed_skeleton_with_mot = {}
#     ids = []
#     skeleton = []
#     for tracker_id, kpt_seq in input:
#         ids.append(tracker_id)
#         kpts = np.array(kpt_seq.kpts, dtype=np.float32)[:, :, :2]
#         kpts = np.expand_dims(np.transpose(kpts, [2, 0, 1]),
#                               -1)  #T, K, C -> C, T, K, 1
#         bbox = np.array(kpt_seq.bboxes, dtype=np.float32)
#         skeleton.append(refine_keypoint_coordinary(kpts, bbox, coord_size))
#     parsed_skeleton_with_mot["mot_id"] = ids
#     parsed_skeleton_with_mot["skeleton"] = skeleton
#     return parsed_skeleton_with_mot