import torch  # pytorch 0.4.0! fft
import torch.nn as nn
import time
import scipy.io
import os
from torch import optim
import numpy as np
from dcf.train.util import *
from collections import OrderedDict


def complex_mul(x, z):
    out_real = x[..., 0] * z[..., 0] - x[..., 1] * z[..., 1]
    out_imag = x[..., 0] * z[..., 1] + x[..., 1] * z[..., 0]
    return torch.stack((out_real, out_imag), -1)


def complex_mulconj(x, z):
    out_real = x[..., 0] * z[..., 0] + x[..., 1] * z[..., 1]
    out_imag = x[..., 1] * z[..., 0] - x[..., 0] * z[..., 1]
    return torch.stack((out_real, out_imag), -1)


class DCFNetFeature(nn.Module):
    def __init__(self, model_path, layer=3):
        super(DCFNetFeature, self).__init__()
        assert layer==2 or layer==3, 'invalid layers'
        self.model_path = model_path
        self.layer = layer
        ################# RGB #################
        self.RGB_feature1 = nn.Sequential(
            OrderedDict([('RGB1',
                          nn.Sequential(
                            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(96),
                            nn.Dropout(0.5),
                            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))

        self.RGB_feature2 = nn.Sequential(
            OrderedDict([('RGB2',
                          nn.Sequential(
                              nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=3, dilation=3),
                              nn.ReLU(),
                              nn.BatchNorm2d(256),
                              nn.Dropout(0.5),
                              nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))

        if layer == 3:
            self.RGB_feature3 = nn.Sequential(
                OrderedDict([('RGB3',
                              nn.Sequential(
                                  nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3, dilation=3),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(512),
                                  nn.Dropout(0.5),
                                  nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                                )
                              )]))

        ################# Thermal #################
        self.T_feature1 = nn.Sequential(
            OrderedDict([('T1',
                          nn.Sequential(
                              nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.BatchNorm2d(96),
                              nn.Dropout(0.5),
                              nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))

        self.T_feature2 = nn.Sequential(
            OrderedDict([('T2',
                          nn.Sequential(
                              nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=3, dilation=3),
                              nn.ReLU(),
                              nn.BatchNorm2d(256),
                              nn.Dropout(0.5),
                              nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))

        if layer == 3:
            self.T_feature3 = nn.Sequential(
                OrderedDict([('T3',
                              nn.Sequential(
                                  nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=3, dilation=3),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(512),
                                  nn.Dropout(0.5),
                                  nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                                )
                              )]))

        ################# Share #################
        self.Share_feature1 = nn.Sequential(
            OrderedDict([('Share1',
                          nn.Sequential(
                            nn.Conv2d(3, 96, kernel_size=7, stride=1, padding=3),
                            nn.ReLU(),
                            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))

        self.Share_feature2 = nn.Sequential(
            OrderedDict([('Share2',
                          nn.Sequential(
                            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=4, dilation=2),
                            nn.ReLU(),
                            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
                            )
                          )]))
        if layer == 3:
            self.Share_feature3 = nn.Sequential(
                OrderedDict([('Share3',
                              nn.Sequential(
                                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2, dilation=2),
                                nn.ReLU(),
                                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
                                )
                              )]))

        if layer == 3:
            self.fusion_gate = nn.Sequential(
                OrderedDict([('fusion_gate',
                              nn.Sequential(
                                  nn.Conv2d(1024, 512, kernel_size=1, stride=1),
                                  nn.Sigmoid()
                                )
                              )]))
        elif layer == 2:
            self.fusion_gate = nn.Sequential(
                OrderedDict([('fusion_gate',
                              nn.Sequential(
                                  nn.Conv2d(512, 256, kernel_size=1, stride=1),
                                  nn.Sigmoid()
                              )
                              )]))

        self.build_param_dict()
        if os.path.splitext(self.model_path)[1] == '.mat':
            #assert self.stage == 'GA', 'load module error'
            self.load_mat_model(self.model_path)

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params

    def append_params(self, params, module, prefix):
        for child in module.children():
            for k, p in child._parameters.items():
                if p is None: continue

                if isinstance(child, nn.BatchNorm2d):
                    name = prefix + '_bn_' + k
                else:
                    name = prefix + '_' + k

                if name not in params:
                    params[name] = p
                else:
                    raise RuntimeError("Duplicated param name: %s" % (name))

    def build_param_dict(self):
        self.params = OrderedDict()

        # **********************RGB*************************************
        for name, module in self.RGB_feature1.named_children():
            self.append_params(self.params, module, name)

        for name, module in self.RGB_feature2.named_children():
            self.append_params(self.params, module, name)

        if self.layer == 3:
            for name, module in self.RGB_feature3.named_children():
                self.append_params(self.params, module, name)

        # **********************T*************************************
        for name, module in self.T_feature1.named_children():
            self.append_params(self.params, module, name)

        for name, module in self.T_feature2.named_children():
            self.append_params(self.params, module, name)

        if self.layer == 3:
            for name, module in self.T_feature3.named_children():
                self.append_params(self.params, module, name)

        # **********************Share*************************************
        for name, module in self.Share_feature1.named_children():
            self.append_params(self.params, module, name)

        for name, module in self.Share_feature2.named_children():
            self.append_params(self.params, module, name)

        if self.layer == 3:
            for name, module in self.Share_feature3.named_children():
                self.append_params(self.params, module, name)

        # **********************FusionGate*************************************
        for name, module in self.fusion_gate.named_children():
            self.append_params(self.params, module, name)

        # print(self.params.keys())

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights

        weight, bias = mat_layers[0 * 4]['weights'].item()[0]
        self.Share_feature1[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.Share_feature1[0][0].bias.data = torch.from_numpy(bias[:, 0])

        weight, bias = mat_layers[1 * 4]['weights'].item()[0]
        self.Share_feature2[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.Share_feature2[0][0].bias.data = torch.from_numpy(bias[:, 0])

        if self.layer == 3:
            weight, bias = mat_layers[2 * 4]['weights'].item()[0]
            self.Share_feature3[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.Share_feature3[0][0].bias.data = torch.from_numpy(bias[:, 0])

        print('load mat finish!')

    def forward(self, xr=None, xt=None):

        # **********************Layer1*************************
        feat1_r = self.RGB_feature1(xr)
        feat1_t = self.T_feature1(xt)

        share1_r = self.Share_feature1(xr)
        share1_t = self.Share_feature1(xt)

        fused1_r = feat1_r + share1_r
        fused1_t = feat1_t + share1_t

        # **********************Layer2*************************
        feat2_r = self.RGB_feature2(fused1_r)
        feat2_t = self.T_feature2(fused1_t)

        share2_r = self.Share_feature2(fused1_r)
        share2_t = self.Share_feature2(fused1_t)

        fused2_r = feat2_r + share2_r
        fused2_t = feat2_t + share2_t

        if self.layer == 2:
            cat_fused = torch.cat((fused2_r, fused2_t), dim=1)
            gated_weight = self.fusion_gate(cat_fused)
            output = (gated_weight) * fused2_r + (1-gated_weight) * fused2_t
        elif self.layer == 3:
        # **********************Layer3*************************
            feat3_r = self.RGB_feature3(fused2_r)
            feat3_t = self.T_feature3(fused2_t)

            share3_r = self.Share_feature3(fused2_r)
            share3_t = self.Share_feature3(fused2_t)

            fused3_r = feat3_r + share3_r
            fused3_t = feat3_t + share3_t
            cat_fused = torch.cat((fused3_r, fused3_t), dim=1)
            gated_weight = self.fusion_gate(cat_fused)
            output = (gated_weight) * fused3_r + (1-gated_weight) * fused3_t

        # print(output.shape)
        #return self.feature(x)
        return output


class DCFNet(nn.Module):
    def __init__(self, config=None):
        super(DCFNet, self).__init__()
        self.feature = DCFNetFeature(config.feature_path, config.layer)
        self.yf = config.yf.clone()
        self.lambda0 = config.lambda0

    def forward(self, zr, zt, xr, xt):
        z = self.feature(zr, zt)
        x = self.feature(xr, xt)
        zf = torch.rfft(z, signal_ndim=2)
        xf = torch.rfft(x, signal_ndim=2)

        kzzf = torch.sum(torch.sum(zf ** 2, dim=4, keepdim=True), dim=1, keepdim=True)
        kxzf = torch.sum(complex_mulconj(xf, zf), dim=1, keepdim=True)
        alphaf = self.yf.to(device=z.device) / (kzzf + self.lambda0)  # very Ugly
        response = torch.irfft(complex_mul(kxzf, alphaf), signal_ndim=2)
        return response

def set_optimizer(model, opts):
    params = model.get_learnable_params()
    momentum = opts.momentum
    lr_base = opts.lr
    lr_mult = opts.lr_mult
    w_decay = opts.w_decay
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer

from dcf.train.config import TrackerConfig
if __name__ == '__main__':
    # feature = DCFNetFeature(layer=2).cuda()
    config = TrackerConfig()
    print(config.output_sigma)
    net = DCFNet(config)
    optimizer = set_optimizer(net.feature, config)
    #print(net.feature.get_learnable_params().keys())
    net = net.cuda()
    xr = torch.rand((1, 3, 125, 125), dtype=torch.float).cuda()
    xt = torch.rand((1, 3, 125, 125), dtype=torch.float).cuda()
    zr = torch.rand((1, 3, 125, 125), dtype=torch.float).cuda()
    zt = torch.rand((1, 3, 125, 125), dtype=torch.float).cuda()
    time1 = time.time()
    y = net(xr, xt, zr, zt)
    time2 = time.time()
    print(time2 - time1)
    # network test
    # net = DCFNet()
    # net.eval()



