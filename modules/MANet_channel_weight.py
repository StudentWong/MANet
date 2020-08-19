import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor


    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


def append_params(params, module, prefix):
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


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, stage, model_path1=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.stage = stage
        self.dim_weight1 = nn.Parameter(torch.tensor(np.zeros(192), dtype=torch.float), requires_grad=True)
        self.dim_weight2 = nn.Parameter(torch.tensor(np.zeros(512), dtype=torch.float), requires_grad=True)
        self.dim_weight3 = nn.Parameter(torch.tensor(np.zeros(1024), dtype=torch.float), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        # ****************RGB_para****************
        self.RGB_para1_3x3 = nn.Sequential(OrderedDict([
            ('Rconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(96),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2)
                                     ))]))
        self.RGB_VGG1 = nn.Sequential(OrderedDict([
            ('RVGG1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)))]))

        self.RGB_para2_1x1 = nn.Sequential(OrderedDict([
            ('Rconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))

             )]))

        self.RGB_VGG2 = nn.Sequential(OrderedDict([
            ('RVGG2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)))]))

        self.RGB_para3_1x1 = nn.Sequential(OrderedDict([
            ('Rconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(512),
                                     nn.Dropout(0.5),
                                     LRN()
                                     )

             )]))

        self.RGB_VGG3 = nn.Sequential(OrderedDict([
            ('RVGG3',nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU()))]))

        # *********T_para**********************
        self.T_para1_3x3 = nn.Sequential(OrderedDict([
            ('Tconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(96),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))
             )]))

        self.T_VGG1 = nn.Sequential(OrderedDict([
            ('TVGG1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)))]))

        self.T_para2_1x1 = nn.Sequential(OrderedDict([
            ('Tconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2))

             )]))

        self.T_VGG2 = nn.Sequential(OrderedDict([
            ('TVGG2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2)))]))

        self.T_para3_1x1 = nn.Sequential(OrderedDict([
            ('Tconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2),

                                     nn.ReLU(),
                                     nn.BatchNorm2d(512),
                                     nn.Dropout(0.5),
                                     LRN()
                                     )

             )]))

        self.T_VGG3 = nn.Sequential(OrderedDict([
            ('TVGG3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU()))]))

        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(1024 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))

        self.layers_small1 = nn.Sequential(OrderedDict([
            ('conv1_small', nn.Sequential(nn.Conv2d(3, 96, kernel_size=3, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(96),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2)))]))

        self.layers_small2 = nn.Sequential(OrderedDict([
            ('conv2_small', nn.Sequential(nn.Conv2d(96, 256, kernel_size=1, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     LRN(),
                                     nn.MaxPool2d(kernel_size=5, stride=2)))]))

        self.layers_small3 = nn.Sequential(OrderedDict([
            ('conv3_small', nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2),
                                     nn.ReLU(),
                                     nn.BatchNorm2d(512),
                                     nn.Dropout(0.5),
                                     LRN()))]))

        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        if model_path1 is not None:
            if os.path.splitext(model_path1)[1] == '.pth':
                assert self.stage == 'MA' or self.stage == 'IA', 'load module error'
                self.load_model(model_path1)
            elif os.path.splitext(model_path1)[1] == '.mat':
                assert self.stage == 'GA', 'load module error'
                self.load_mat_model(model_path1)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path1))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        self.params['dim_weight1'] = self.dim_weight1
        self.params['dim_weight2'] = self.dim_weight2
        self.params['dim_weight3'] = self.dim_weight3

        # **********************RGB*************************************
        for name, module in self.RGB_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_VGG1.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_VGG2.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_para3_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.RGB_VGG3.named_children():
            append_params(self.params, module, name)

        # **********************T*************************************
        for name, module in self.T_para1_3x3.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_VGG1.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para2_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_VGG2.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_para3_1x1.named_children():
            append_params(self.params, module, name)

        for name, module in self.T_VGG3.named_children():
            append_params(self.params, module, name)

        # **********************conv*fc*************************************
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d' % (k))

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

    def forward(self, xR=None, xT=None, feat=None, k=0, in_layer='conv1', out_layer='fc6'):

        run = False

        for name, module in self.layers.named_children():
            if name == in_layer:
                run = True
            if run:
                # print(name)
                if name == 'conv1':
                    
                    feat_T = self.T_para1_3x3(xT)
                    feat_R = self.RGB_para1_3x3(xR)
                    VGG_T = self.T_VGG1(xT)
                    VGG_R = self.RGB_VGG1(xR)

                    VGG_MT = module(xT)
                    feat_MT = self.layers_small1(xT)
                    VGG_MR = module(xR)
                    feat_MR = self.layers_small1(xR)

                    cat_T = torch.cat((feat_T, VGG_T), 1)
                    cat_R = torch.cat((feat_R, VGG_R), 1)
                    cat_MT = torch.cat((feat_MT, VGG_MT), 1)
                    cat_MR = torch.cat((feat_MR, VGG_MR), 1)

                    weight_M1 = self.sigmoid(self.dim_weight1)
                    weight_TR1 = 1-weight_M1
                    weight_M1 = weight_M1.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    weight_TR1 = weight_TR1.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    # print(weight_M1.shape)
                    # print(weight_TR1.shape)

                    featT = weight_TR1.expand_as(cat_T) * cat_T + weight_M1.expand_as(cat_MT) * cat_MT
                    featR = weight_TR1.expand_as(cat_R) * cat_R + weight_M1.expand_as(cat_MR) * cat_MR

                    # exit()

                if name == 'conv2':
                    feat_T_split, VGG_T_split = featT.split([96, 96], 1)
                    feat_R_split, VGG_R_split = featR.split([96, 96], 1)

                    feat_T = self.T_para2_1x1(feat_T_split)
                    feat_R = self.RGB_para2_1x1(feat_R_split)
                    VGG_T = self.T_VGG2(VGG_T_split)
                    VGG_R = self.RGB_VGG2(VGG_R_split)

                    VGG_MT = module(VGG_T_split)
                    feat_MT = self.layers_small2(feat_T_split)
                    VGG_MR = module(VGG_R_split)
                    feat_MR = self.layers_small2(feat_R_split)

                    cat_T = torch.cat((feat_T, VGG_T), 1)
                    cat_R = torch.cat((feat_R, VGG_R), 1)
                    cat_MT = torch.cat((feat_MT, VGG_MT), 1)
                    cat_MR = torch.cat((feat_MR, VGG_MR), 1)

                    weight_M2 = self.sigmoid(self.dim_weight2)
                    weight_TR2 = 1 - weight_M2
                    weight_M2 = weight_M2.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    weight_TR2 = weight_TR2.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    # print(weight_M1.shape)
                    # print(weight_TR1.shape)
                    featT = weight_TR2.expand_as(cat_T) * cat_T + weight_M2.expand_as(cat_MT) * cat_MT
                    featR = weight_TR2.expand_as(cat_R) * cat_R + weight_M2.expand_as(cat_MR) * cat_MR


                    # feat_T = self.T_para2_1x1(featT)
                    # feat_R = self.RGB_para2_1x1(featR)
                    # feat_MT = module(featT)
                    # feat_MR = module(featR)
                    #
                    # featR = feat_MR + feat_R
                    # featT = feat_MT + feat_T

                if name == 'conv3':
                    feat_T_split, VGG_T_split = featT.split([256, 256], 1)
                    feat_R_split, VGG_R_split = featR.split([256, 256], 1)

                    feat_T = self.T_para3_1x1(feat_T_split)
                    feat_R = self.RGB_para3_1x1(feat_R_split)
                    VGG_T = self.T_VGG3(VGG_T_split)
                    VGG_R = self.RGB_VGG3(VGG_R_split)

                    VGG_MT = module(VGG_T_split)
                    feat_MT = self.layers_small3(feat_T_split)
                    VGG_MR = module(VGG_R_split)
                    feat_MR = self.layers_small3(feat_R_split)

                    cat_T = torch.cat((feat_T, VGG_T), 1)
                    cat_R = torch.cat((feat_R, VGG_R), 1)
                    cat_MT = torch.cat((feat_MT, VGG_MT), 1)
                    cat_MR = torch.cat((feat_MR, VGG_MR), 1)

                    weight_M3 = self.sigmoid(self.dim_weight3)
                    weight_TR3 = 1 - weight_M3
                    weight_M3 = weight_M3.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    weight_TR3 = weight_TR3.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    # print(weight_M1.shape)
                    # print(weight_TR1.shape)
                    featT = weight_TR3.expand_as(cat_T) * cat_T + weight_M3.expand_as(cat_MT) * cat_MT
                    featR = weight_TR3.expand_as(cat_R) * cat_R + weight_M3.expand_as(cat_MR) * cat_MR

                    feat = featT + featR
                    # feat_T = self.T_para3_1x1(featT)
                    # feat_R = self.RGB_para3_1x1(featR)
                    # feat_MT = module(featT)
                    # feat_MR = module(featR)
                    #
                    # featR = feat_MR + feat_R
                    # featT = feat_MT + feat_T
                    # feat = torch.cat((featR, featT), 1)  # train 1:1   #test 1:1.4

                    feat = feat.view(feat.size(0), -1)
                if name == 'fc4':
                    feat = module(feat)

                if name == 'fc5':
                    feat = module(feat)

                if name == out_layer:
                    return feat

        feat = self.branches[k](feat)
        if out_layer == 'fc6':
            return feat
        elif out_layer == 'fc6_softmax':
            return F.softmax(feat, dim=1)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']

        self.dim_weight1 = states['dim_weight1']
        self.dim_weight2 = states['dim_weight2']
        self.dim_weight3 = states['dim_weight3']
        # print(shared_layers.keys())
        # print(self.layers)
        # exit()
        if self.stage == 'MA':
            conv1 = OrderedDict()
            conv1['0.weight'] = states['shared_layers']['conv1.0.weight']
            conv1['0.bias'] = states['shared_layers']['conv1.0.bias']
            self.layers.conv1.load_state_dict(conv1)

            conv2 = OrderedDict()
            conv2['0.weight'] = states['shared_layers']['conv2.0.weight']
            conv2['0.bias'] = states['shared_layers']['conv2.0.bias']
            self.layers.conv2.load_state_dict(conv2)

            conv3 = OrderedDict()
            conv3['0.weight'] = states['shared_layers']['conv3.0.weight']
            conv3['0.bias'] = states['shared_layers']['conv3.0.bias']
            self.layers.conv3.load_state_dict(conv3)

        elif self.stage == 'IA':
            self.layers.load_state_dict(shared_layers)

        para1_small_layer = states['layers_small1']
        self.layers_small1.load_state_dict(para1_small_layer, strict=True)
        para2_small_layer = states['layers_small2']
        self.layers_small2.load_state_dict(para2_small_layer, strict=True)
        para3_small_layer = states['layers_small3']
        self.layers_small3.load_state_dict(para3_small_layer, strict=True)

        if self.stage == 'MA':
            pass
        elif self.stage == 'IA':
            para1_layers = states['RGB_para1_3x3']
            self.RGB_para1_3x3.load_state_dict(para1_layers, strict=True)
            para2_layers = states['RGB_para2_1x1']
            self.RGB_para2_1x1.load_state_dict(para2_layers, strict=True)
            para3_layers = states['RGB_para3_1x1']
            self.RGB_para3_1x1.load_state_dict(para3_layers, strict=True)

            para1_layers = states['T_para1_3x3']
            self.T_para1_3x3.load_state_dict(para1_layers, strict=True)
            para2_layers = states['T_para2_1x1']
            self.T_para2_1x1.load_state_dict(para2_layers, strict=True)
            para3_layers = states['T_para3_1x1']
            self.T_para3_1x1.load_state_dict(para3_layers, strict=True)

            para1_layers = states['T_VGG1']
            self.T_VGG1.load_state_dict(para1_layers, strict=True)
            para2_layers = states['T_VGG2']
            self.T_VGG2.load_state_dict(para2_layers, strict=True)
            para3_layers = states['T_VGG3']
            self.T_VGG3.load_state_dict(para3_layers, strict=True)

            para1_layers = states['RGB_VGG1']
            self.RGB_VGG1.load_state_dict(para1_layers, strict=True)
            para2_layers = states['RGB_VGG2']
            self.RGB_VGG2.load_state_dict(para2_layers, strict=True)
            para3_layers = states['RGB_VGG3']
            self.RGB_VGG3.load_state_dict(para3_layers, strict=True)


        print('load pth finish!')

    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])
            if i == 0:
                self.T_VGG1[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.T_VGG1[0][0].bias.data = torch.from_numpy(bias[:, 0])
            elif i == 1:
                self.T_VGG2[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.T_VGG2[0][0].bias.data = torch.from_numpy(bias[:, 0])
            elif i == 2:
                self.T_VGG3[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
                self.T_VGG3[0][0].bias.data = torch.from_numpy(bias[:, 0])
        print('load mat finish!')


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        # return prec.data[0]
        return prec.item()


from options.options_GA_dim import opts as opts_GA
from options.options_MA_dim import opts as opts_MA
import torch.optim as optim

if __name__ == '__main__':
    net = MDNet(opts_MA['stage'], 'logs/GA_dim.pth', 1).cuda()
    print(net.dim_weight1)
    print(net.dim_weight2)
    print(net.dim_weight3)
    # for name in net.state_dict():
    #     print(name)
    # optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    xr = torch.rand((5, 3, 107, 107), dtype=torch.float).cuda()
    xt = torch.rand((5, 3, 107, 107), dtype=torch.float).cuda()
    out = net(xR=xr, xT=xt, out_layer='fc6_softmax')
    loss = out - torch.tensor([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=torch.float).cuda()
    loss = loss.mean()
    loss.backward()
    print(out)
