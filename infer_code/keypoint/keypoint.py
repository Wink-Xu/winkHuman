# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import time
import yaml
import glob
from functools import reduce

from PIL import Image
import cv2
import math
import numpy as np


import sys
# add deploy path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'])))
sys.path.insert(0, parent_path)

from preprocess import preprocess, NormalizeImage, Permute
from .keypoint_process import get_affine_transform, get_final_preds, get_max_preds, save_debug_images
from visualize import visualize_pose

import torchvision.transforms as transforms
from .models.pose_hrnet import get_pose_net
from .config import cfg
from .config import update_config
from utils import argsparser, Timer

class KeyPointDetector(object):
    """
    Args:
        model_dir (str): root path of model.pdiparams, model.pdmodel and infer_cfg.yml
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16)
        batch_size (int): size of pre batch in inference
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        cpu_threads (int): cpu threads
        enable_mkldnn (bool): whether to open MKLDNN
        use_dark(bool): whether to use postprocess in DarkPose
    """

    def __init__(self,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False,
                 output_dir='output',
                 threshold=0.5,
                 use_dark=True):
        self.cfg = self.set_config(model_dir)     
        self.model = get_pose_net(self.cfg, is_train = False)
        self.model.load_state_dict(torch.load(self.cfg.TEST.MODEL_FILE, strict =False))
        self.model = torch.nn.DataParallel(self.model, device_ids=self.cfg.GPUS).cuda()
        self.batch_size = batch_size
        self.det_times = Timer()
        self.use_dark = use_dark

    def set_config(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        arch = yml_conf['arch']
        config_path = yml_conf['config_path']
        self.cfg = update_config(cfg, config_path)
        print('-----------Keypoint  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', arch))
        print('%s: %s' % ('Config Path', config_path))
        print('--------------------------------------------')        
        return 

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale = np.array(
            [w * 1.0 / 200, h * 1.0 / 200],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def merge_batch_result(self, results):
        if len(results) == 1:
            return results[0]
        return np.vstack(results)
        
    def preprocess(self, inputs):
        # Data loading code
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        self.c, self.s = self._xywh2cs(0, 0, inputs.shape[2], inputs.shape[1])
        r = 0
        
        image_size = np.array(self.cfg.MODEL.IMAGE_SIZE)
        trans = get_affine_transform(c, s, r, image_size)
        inputs = cv2.warpAffine(
            data_numpy,
            trans,
            (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)

        inputs = transform(inputs)
        return inputs 

    def postprocess(self, inputs, result):
        np_heatmap = result['heatmap']
        np_masks = result['masks']
        # postprocess output of predictor
        idx = 0
        num_samples = inputs.shape[0]
        all_preds = np.zeros(
            (num_samples, self.cfg.MODEL.NUM_JOINTS, 3),
            dtype=np.float32
        )
    
        preds, maxvals = get_final_preds(
            self.cfg, result.clone().cpu().numpy(), self.c, self.s)
        pred, _ = get_max_preds(result.clone().cpu().numpy())

        all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
        all_preds[idx:idx + num_images, :, 2:3] = maxvals
           

        return all_preds, pred

    def predict_image(self,
                      image_list,
                      run_benchmark=False,
                      repeats=1,
                      visual=True):
        results = []
        batch_loop_cnt = math.ceil(float(len(image_list)) / self.batch_size)
        for i in range(batch_loop_cnt):
            start_index = i * self.batch_size
            end_index = min((i + 1) * self.batch_size, len(image_list))
            batch_image_list = image_list[start_index:end_index]

                # preprocess
            self.det_times.preprocess_time_s.start()
            inputs = self.preprocess(batch_image_list)
            self.det_times.preprocess_time_s.end()

            # model prediction
            self.det_times.inference_time_s.start()
            outputs = self.model(inputs)
            self.det_times.inference_time_s.end()
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs
            # postprocess
            self.det_times.postprocess_time_s.start()
            result, pred = self.postprocess(inputs, output)
            self.det_times.postprocess_time_s.end()
            self.det_times.img_num += len(batch_image_list)

            if visual:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                save_debug_images(self.cfg, inputs,  pred*4, output, 
                                    self.output_dir)

            results.append(result)
            if visual:
                print('Test iter {}'.format(i))
        results = self.merge_batch_result(results)
        return results


def create_inputs(imgs, im_info):
    """generate input for different model type
    Args:
        imgs (list(numpy)): list of image (np.ndarray)
        im_info (list(dict)): list of image info
    Returns:
        inputs (dict): input of model
    """
    inputs = {}
    inputs['image'] = np.stack(imgs, axis=0).astype('float32')
    im_shape = []
    for e in im_info:
        im_shape.append(np.array((e['im_shape'])).astype('float32'))
    inputs['im_shape'] = np.stack(im_shape, axis=0)
    return inputs


class PredictConfig_KeyPoint():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.archcls = KEYPOINT_SUPPORT_MODELS[yml_conf['arch']]
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.tagmap = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'keypoint_bottomup' == self.archcls:
            self.tagmap = True
        self.print_config()

    def check_model(self, yml_conf):
        """
        Raises:
            ValueError: loaded model not in supported model type 
        """
        for support_model in KEYPOINT_SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
            'arch'], KEYPOINT_SUPPORT_MODELS))

    def print_config(self):
        print('-----------  Model Configuration -----------')
        print('%s: %s' % ('Model Arch', self.arch))
        print('%s: ' % ('Transform Order'))
        for op_info in self.preprocess_infos:
            print('--%s: %s' % ('transform op', op_info['type']))
        print('--------------------------------------------')


def visualize(image_list, results, visual_thresh=0.6, save_dir='output'):
    im_results = {}
    for i, image_file in enumerate(image_list):
        skeletons = results['keypoint']
        scores = results['score']
        skeleton = skeletons[i:i + 1]
        score = scores[i:i + 1]
        im_results['keypoint'] = [skeleton, score]
        visualize_pose(
            image_file,
            im_results,
            visual_thresh=visual_thresh,
            save_dir=save_dir)


def main():
    detector = KeyPointDetector(
        FLAGS.model_dir,
        device=FLAGS.device,
        run_mode=FLAGS.run_mode,
        batch_size=FLAGS.batch_size,
        trt_min_shape=FLAGS.trt_min_shape,
        trt_max_shape=FLAGS.trt_max_shape,
        trt_opt_shape=FLAGS.trt_opt_shape,
        trt_calib_mode=FLAGS.trt_calib_mode,
        cpu_threads=FLAGS.cpu_threads,
        enable_mkldnn=FLAGS.enable_mkldnn,
        threshold=FLAGS.threshold,
        output_dir=FLAGS.output_dir,
        use_dark=FLAGS.use_dark)

    dataset = LoadImages(FLAGS.image_file)
    img_list = get_test_images(FLAGS.image_dir, FLAGS.image_file)
    detector.predict_image(img_list, FLAGS.run_benchmark, repeats=10)
    
    detector.det_times.info(average=True)


if __name__ == '__main__':
    parser = argsparser()
    FLAGS = parser.parse_args()
    print_arguments(FLAGS)

    main()
