import torch.utils.data as data
from os.path import join
from os import listdir
import dcf.train.util as util
import time
import os
import torch
import cv2
import json
import numpy as np


class VID(data.Dataset):
    def __init__(self, file='/home/studentw/disk3/tracker/RGB_T234/rgbt234.txt',
                 root='/home/studentw/disk3/tracker/RGB_T234/',
                 range=20, data='RGBT234', padding=2, fixsize=125,
                 output_sigma=4.167, flip=True, bias=False):
        self.dataset = data
        self.padding = padding
        self.fixsize = fixsize
        self.output_sigma = output_sigma
        self.flip = flip
        self.bias = bias
        # print(file)
        with open(file, 'r') as fp_txt:
            self.sequences = fp_txt.readlines()

        if self.dataset == 'RGBT234':
            self.gt_name = 'init.txt'
            self.infrared_folder = 'infrared'
            self.rgb_folder = 'visible'

        self.seqs_inf = dict()
        for i, seq_name in enumerate(self.sequences):
            seq_inf = dict()
            if seq_name.endswith('\n'):
                self.sequences[i] = seq_name[:-1]

            seq_rgb = sorted(listdir(join(root, self.sequences[i], self.rgb_folder)))
            seq_infrared = sorted(listdir(join(root, self.sequences[i], self.infrared_folder)))
            with open(join(root, self.sequences[i], self.gt_name), 'r') as fp_gt:
                gt = fp_gt.readlines()
            assert len(seq_rgb) == len(seq_infrared) and len(gt)
            seq_len = len(seq_rgb)
            seq_inf['seq_rgb'] = seq_rgb
            seq_inf['seq_infrared'] = seq_infrared
            seq_inf['seq_len'] = seq_len
            seq_inf['gt_str'] = gt

            self.seqs_inf[self.sequences[i]] = seq_inf

        self.sequences = sorted(self.sequences)
        # print(len(self.seqs_inf))
        # print(self.sequences)
        self.len = len(self.sequences)
        self.root = root
        self.range = range
        self.mean_RGB = np.expand_dims(np.expand_dims(np.array([109, 120, 119]), axis=1), axis=1).astype(np.float32)
        self.mean_T = np.expand_dims(np.expand_dims(np.array([128, 128, 128]), axis=1), axis=1).astype(np.float32)
        self.pointer = [0] * self.len

    def __getitem__(self, item):
        seq = self.sequences[item]
        seq_inf = self.seqs_inf[seq]
        temp_id = self.pointer[item]
        search_id = min(temp_id + np.random.randint(1, self.range+1), seq_inf['seq_len'] - 1)


        template_rgb = cv2.imread(join(self.root, seq, self.rgb_folder, seq_inf['seq_rgb'][temp_id]))
        template_infrared = cv2.imread(join(self.root, seq, self.infrared_folder, seq_inf['seq_infrared'][temp_id]))

        search_rgb = cv2.imread(join(self.root, seq, self.rgb_folder, seq_inf['seq_rgb'][search_id]))
        search_infrared = cv2.imread(join(self.root, seq, self.infrared_folder, seq_inf['seq_infrared'][search_id]))

        template_gt_str = seq_inf['gt_str'][temp_id][:-1] if seq_inf['gt_str'][temp_id].endswith('\n') else seq_inf['gt_str'][temp_id]
        search_gt_str = seq_inf['gt_str'][search_id][:-1] if seq_inf['gt_str'][search_id].endswith('\n') else seq_inf['gt_str'][search_id]

        template_gt_str_split = template_gt_str.split(',')
        search_gt_str_split = search_gt_str.split(',')


        template_gt_cwh = [int(template_gt_str_split[0])+int(template_gt_str_split[2])/2, int(template_gt_str_split[1])+int(template_gt_str_split[3])/2,
                              int(template_gt_str_split[2]), int(template_gt_str_split[3])]
        search_gt_cwh = [int(search_gt_str_split[0])+int(search_gt_str_split[2])/2, int(search_gt_str_split[1])+int(search_gt_str_split[3])/2,
                              int(search_gt_str_split[2]), int(search_gt_str_split[3])]

        template_gt_cwh = np.array(template_gt_cwh, dtype=np.float)
        search_gt_cwh = np.array(search_gt_cwh, dtype=np.float)


        # template_gt_corner = util.cwh2corner(template_gt_cwh)
        # search_gt_corner = util.cwh2corner(search_gt_cwh)
        template_gt_cwh_with_padding = template_gt_cwh
        search_gt_cwh_with_padding = search_gt_cwh
        template_gt_cwh_with_padding[2:4] = template_gt_cwh_with_padding[2:4] * (self.padding/2 + 1)
        search_gt_cwh_with_padding[2:4] = search_gt_cwh_with_padding[2:4] * (self.padding/2 + 1)

        template_rgb_region = util.crop_chw(template_rgb, util.cwh2corner(template_gt_cwh_with_padding), self.fixsize)
        template_infrared_region = util.crop_chw(template_infrared, util.cwh2corner(template_gt_cwh_with_padding), self.fixsize)

        search_rgb_region = util.crop_chw(search_rgb, util.cwh2corner(search_gt_cwh_with_padding), self.fixsize)
        search_infrared_region = util.crop_chw(search_infrared, util.cwh2corner(search_gt_cwh_with_padding), self.fixsize)

        search_rgb_region = search_rgb_region - self.mean_RGB
        template_rgb_region = template_rgb_region - self.mean_RGB
        search_infrared_region = search_infrared_region - self.mean_T
        template_infrared_region = template_infrared_region - self.mean_T

        # print(search_rgb_region.shape)
        # cv2.imshow("1", search_rgb_region.transpose((1, 2, 0)))
        # cv2.imshow("2", search_infrared_region.transpose((1, 2, 0)))
        # cv2.imshow("3", template_rgb_region.transpose((1, 2, 0)))
        # cv2.imshow("4", template_infrared_region.transpose((1, 2, 0)))
        # cv2.waitKey(0)
        # print(template_gt)
        # print(search_gt)

        # crop_chw
        # print(join(self.root, seq, self.rgb_folder, seq_inf['seq_rgb'][temp_id]))
        # print(join(self.root, seq, self.rgb_folder, seq_inf['seq_infrared'][temp_id]))
        # template_rgb = cv2.rectangle(template_rgb, (int(template_gt[0]), int(template_gt[1])),
        #                              (int(template_gt[0] + template_gt[2]), int(template_gt[1] + template_gt[3])),
        #                              (255, 0, 0), thickness=1)
        # search_rgb = cv2.rectangle(search_rgb, (int(search_gt[0]), int(search_gt[1])),
        #                              (int(search_gt[0] + search_gt[2]), int(search_gt[1] + search_gt[3])),
        #                              (255, 0, 0), thickness=1)
        # template_infrared = cv2.rectangle(template_infrared, (int(template_gt[0]), int(template_gt[1])),
        #                              (int(template_gt[0] + template_gt[2]), int(template_gt[1] + template_gt[3])),
        #                              (255, 0, 0), thickness=1)
        # search_infrared = cv2.rectangle(search_infrared, (int(search_gt[0]), int(search_gt[1])),
        #                            (int(search_gt[0] + search_gt[2]), int(search_gt[1] + search_gt[3])),
        #                            (255, 0, 0), thickness=1)
        #

        self.pointer[item] = self.pointer[item] + 1
        if self.pointer[item] > seq_inf['seq_len'] - 2:
            self.pointer[item] = 0


        if self.flip:
            if np.random.rand()>0.5:
                search_rgb_region = np.flip(search_rgb_region, 2)
                search_infrared_region = np.flip(search_infrared_region, 2)
                template_rgb_region = np.flip(template_rgb_region, 2)
                template_infrared_region = np.flip(template_infrared_region, 2)

        return template_rgb_region.astype(np.float32), template_infrared_region.astype(np.float32), \
               search_rgb_region.astype(np.float32), search_infrared_region.astype(np.float32), \
               util.gaussian_shaped_labels(self.output_sigma, [self.fixsize, self.fixsize]).astype(np.float32)
    #print(y.shape)



    def __len__(self):
        return self.len

if __name__ == '__main__':
    data = VID()


    for i in range(0, len(data)):
        sr, si, tr, ti, res = data[i]
        cv2.imshow("1", sr.transpose((1, 2, 0)))
        cv2.waitKey(0)
        cv2.imshow("1", si.transpose((1, 2, 0)))
        cv2.waitKey(0)
        cv2.imshow("1", tr.transpose((1, 2, 0)))
        cv2.waitKey(0)
        cv2.imshow("1", ti.transpose((1, 2, 0)))
        cv2.waitKey(0)
        # print(sr.shape)
        # print(si.shape)
        # print(tr.shape)
        # print(ti.shape)
        print(res)