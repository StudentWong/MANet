from dcf.test.net_test import DCFNet
from dcf.test.config import TrackerConfig
from dcf.test.util import *
import cv2
import numpy as np
import torch
import os.path
import os
import argparse

parser = argparse.ArgumentParser(description='Track DCFNet in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
parser.add_argument('--val_data', default='gtot', type=str, help='val data')
parser.add_argument('--weight', '-w', default='work', type=str, help='directory for saving')
parser.add_argument('--layer', default=2, type=int, help='layer num')
parser.add_argument('--result', '-r', default='dcfresult', type=str, help='directory for saving')

args = parser.parse_args()

print(args)
config = TrackerConfig()
assert args.layer == 2 or args.layer == 3, 'invalid layer'
assert args.val_data=='gtot' or args.val_data=='rgbt234', 'invalid val data'

result_path = os.path.join(config.result_path_root, args.result)
root = config.root

# result_path = os.path.join('/home/htz/ZYDL', args.result)
# root = '/home/htz/ZYDL/MANet'


config.layer = args.layer
config.val_data = args.val_data

if config.val_data == 'rgbt234':
    config.val_file = config.rgbt234_file
    config.val_root = config.rgbt234_root
elif config.val_data == 'gtot':
    config.val_file = config.gtot_file
    config.val_root = config.gtot_root


class DCFNetTraker(object):
    def __init__(self, imr, imt, init_rect, config=TrackerConfig(), gpu=True):
        self.gpu = gpu
        self.config = config
        self.net = DCFNet(config)
        self.net.load_param(config.feature_path)
        self.net.eval()
        self.mean_RGB = np.expand_dims(np.expand_dims(np.array([109, 120, 119]), axis=1), axis=1).astype(np.float32)
        self.mean_T = np.expand_dims(np.expand_dims(np.array([128, 128, 128]), axis=1), axis=1).astype(np.float32)

        if gpu:
            self.net.cuda()

        # confine results
        target_pos, target_sz = rect1_2_cxy_wh(init_rect)
        self.min_sz = np.maximum(config.min_scale_factor * target_sz, 4)
        self.max_sz = np.minimum(imr.shape[:2], config.max_scale_factor * target_sz)

        # crop template
        window_sz = target_sz * (1 + config.padding)
        bbox = cxy_wh_2_bbox(target_pos, window_sz)
        patch_r = crop_chw(imr, bbox, self.config.crop_sz)
        patch_t = crop_chw(imt, bbox, self.config.crop_sz)

        # cv2.imshow('1', patch_r.transpose(1, 2, 0))
        # cv2.waitKey(0)
        # cv2.imshow('1', patch_t.transpose(1, 2, 0))
        # cv2.waitKey(0)

        target_r = patch_r - self.mean_RGB
        target_t = patch_t - self.mean_T

        self.net.update(torch.Tensor(np.expand_dims(target_r, axis=0)).cuda(),
                        torch.Tensor(np.expand_dims(target_t, axis=0)).cuda(), lr=1.0)
        self.target_pos, self.target_sz = target_pos, target_sz
        self.patch_crop_r = np.zeros((config.num_scale, target_r.shape[0], target_r.shape[1], target_r.shape[2]), np.float32)  # buff

        self.patch_crop_t = np.zeros((config.num_scale, target_t.shape[0], target_t.shape[1], target_t.shape[2]),
                                 np.float32)  # buff

    def track(self, imr, imt):
        for i in range(self.config.num_scale):  # crop multi-scale search region
            window_sz = self.target_sz * (self.config.scale_factor[i] * (1 + self.config.padding))
            bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
            self.patch_crop_r[i, :] = crop_chw(imr, bbox, self.config.crop_sz)
            self.patch_crop_t[i, :] = crop_chw(imt, bbox, self.config.crop_sz)

        search_r = self.patch_crop_r - self.mean_RGB
        search_t = self.patch_crop_t - self.mean_T

        if self.gpu:
            response = self.net(torch.Tensor(search_r).cuda(),
                                torch.Tensor(search_t).cuda())
        else:
            response = self.net(torch.Tensor(search_r),
                                torch.Tensor(search_t))

        peak, idx = torch.max(response.view(self.config.num_scale, -1), 1)
        peak = peak.data.cpu().numpy() * self.config.scale_penalties
        best_scale = np.argmax(peak)
        r_max, c_max = np.unravel_index(idx[best_scale].cpu().numpy(),
                                        self.config.net_input_size)

        if r_max > self.config.net_input_size[0] / 2:
            r_max = r_max - self.config.net_input_size[0]
        if c_max > self.config.net_input_size[1] / 2:
            c_max = c_max - self.config.net_input_size[1]
        window_sz = self.target_sz * (self.config.scale_factor[best_scale] * (1 + self.config.padding))

        self.target_pos = self.target_pos + np.array([c_max, r_max]) * window_sz / self.config.net_input_size
        #self.target_sz = np.minimum(np.maximum(window_sz / (1 + self.config.padding), self.min_sz), self.max_sz)
        self.target_sz = window_sz / (1 + self.config.padding)

        # model update
        window_sz = self.target_sz * (1 + self.config.padding)
        bbox = cxy_wh_2_bbox(self.target_pos, window_sz)
        patch_r = crop_chw(imr, bbox, self.config.crop_sz)
        patch_t = crop_chw(imt, bbox, self.config.crop_sz)
        target_r = patch_r - self.mean_RGB
        target_t = patch_t - self.mean_T

        self.net.update(torch.Tensor(np.expand_dims(target_r, axis=0)).cuda(),
                        torch.Tensor(np.expand_dims(target_t, axis=0)).cuda(),
                        lr=self.config.interp_factor)

        return cxy_wh_2_rect1(self.target_pos, self.target_sz)  # 1-index


if __name__ == '__main__':
    # config = TrackerConfig()
    # weight = torch.load(config.feature_path)
    # print(weight['best_loss'])
    # exit()
    config.feature_path = os.path.join(root, args.weight, 'crop_125_2.0', 'model_best.pth.tar')

    if not os.path.exists(result_path):
        os.mkdir(result_path)
    dataset=args.val_data
    with open(config.val_file, 'r') as fp_txt:
        sequences = fp_txt.readlines()
    for i, seq_name in enumerate(sequences):
        if seq_name.endswith('\n'):
            sequences[i] = seq_name[:-1]

    if dataset == 'rgbt234':
        gt_name = 'init.txt'
        infrared_folder = 'infrared'
        rgb_folder = 'visible'

    elif dataset == 'gtot':
        gt_name = 'init.txt'
        infrared_folder = 'i'
        rgb_folder = 'v'

    for seq in sequences:
        gt_init_path = os.path.join(config.val_root, seq, gt_name)
        output_path = 'MANet311-2IC_'+seq+'.txt'
        if not os.path.exists(os.path.join(result_path, output_path)):
            os.mknod(os.path.join(result_path, output_path))


        with open(gt_init_path, 'r') as fp:
            gt_init = fp.readlines()[0]
            if gt_init.endswith('\n'):
                gt_init = gt_init[:-1]
        rgb_path = sorted(os.listdir(os.path.join(config.val_root, seq, rgb_folder)))
        t_path = sorted(os.listdir(os.path.join(config.val_root, seq, infrared_folder)))
        assert len(rgb_path) == len(t_path)
        init_r_img = cv2.imread(os.path.join(config.val_root, seq, rgb_folder, rgb_path[0]))
        init_t_img = cv2.imread(os.path.join(config.val_root, seq, infrared_folder, t_path[0]))
        # print(gt_init)
        gt_split = gt_init.split('\t')

        # print(gt_split)
        # gt_init_rect = [int(gt_split[0])+int(gt_split[2])/2, int(gt_split[1])+int(gt_split[3])/2,
        #                 int(gt_split[2]), int(gt_split[3])]
        gt_init_rect = [int(gt_split[0]) , int(gt_split[1]),
                        int(gt_split[2]), int(gt_split[3])]

        f = open(os.path.join(result_path, output_path), 'w+')

        res = '{} {} {} {} {} {} {} {}'.format(int(gt_split[0]), int(gt_split[1]),

                                               int(gt_split[0])+int(gt_split[2]), int(gt_split[1]),

                                               int(gt_split[0])+int(gt_split[2]),
                                               int(gt_split[1])+int(gt_split[3]),

                                               int(gt_split[0]), int(gt_split[1])+int(gt_split[3])
                                               )
        f.write(res)
        f.write('\n')


        tracker = DCFNetTraker(init_r_img, init_t_img, gt_init_rect, config=config, gpu=True)

        for i in range(1, len(rgb_path)):
            init_r_img = cv2.imread(os.path.join(config.val_root, seq, rgb_folder, rgb_path[i]))
            init_t_img = cv2.imread(os.path.join(config.val_root, seq, infrared_folder, t_path[i]))
            init_r_img_show = init_r_img.copy()
            with torch.no_grad():
                result = tracker.track(init_r_img, init_t_img)
            point1 = (int(result[0]), int(result[1]))
            point2 = (int(result[0]+result[2]), int(result[1]+result[3]))
            # print(point1)
            # print(point2)
            # init_r_img_show = init_r_img_show.copy()
            # imshow = cv2.rectangle(init_r_img_show, point1, point2, (255, 0, 0), thickness=1)
            # cv2.imshow('1', imshow)
            # cv2.waitKey(0)
            # if not os.path.exists(os.path.join(result_path, 'picture')):
            #     os.mkdir(os.path.join(result_path, 'picture'))
            # if not os.path.exists(os.path.join(result_path, 'picture', seq)):
            #     os.mkdir(os.path.join(result_path, 'picture', seq))
            # cv2.imwrite(os.path.join(result_path, 'picture', seq, rgb_path[i]), imshow)
            # print(result)

            res = '{} {} {} {} {} {} {} {}'.format(point1[0],
                                                   point1[1],

                                                   point2[0],
                                                   point1[1],

                                                   point2[0],
                                                   point2[1],

                                                   point1[0],
                                                   point2[1]
                                                   )
            f.write(res)
            f.write('\n')

        f.close()
