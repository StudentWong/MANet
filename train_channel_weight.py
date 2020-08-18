import os
import sys
import argparse
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from pretrain.data_prov import RegionDataset, RegionDataset1
from modules.MANet_channel_weight import *
from options.options_GA_dim import opts as optsGA
from options.options_MA_dim import opts as optsMA
from options.options_IA import opts as optsIA

#********************************************set dataset path ********************************************
#********************************************set seq list .pkl file path  ********************************
# img_home = "/home/studentw/disk3/tracker/RGB_T234/"
img_home = "/home/htz/ZYDL/RGB_T234/"
data_path1 = 'DATA/rgbt234_V.pkl'
# data_path2 = 'DATA/rgbt234_I.pkl'
#*********************************************************************************************************


def set_optimizer(model, opts):
    params = model.get_learnable_params()
    momentum = opts['momentum']
    lr_base = opts['lr']
    lr_mult = opts['lr_mult']
    w_decay = opts['w_decay']
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet(opts):
    
    ## Init dataset ##
    with open(data_path1, 'rb') as fp1:
        data1 = pickle.load(fp1)

    K1 = len(data1)
    dataset1 = [None]*K1
    # print(sorted(data1.items())[0])
    # exit()
    for k, (seqname, seq) in enumerate(sorted(data1.items())):
        # print(seqname)
        img_list1 = seq['images']
        gt1 = seq['gt']
        img_dir1 = os.path.join(img_home, seqname)
        dataset1[k] = RegionDataset(img_dir1, img_list1, gt1, opts)
        # print(dataset1[k])
        # exit()


    with open(data_path2,'rb') as fp2:
        data2=pickle.load(fp2)

    K2=len(data2)
    dataset2=[None]*K2
    for k, (seqname, seq) in enumerate(sorted(data2.items())):
        # print(seqname)
        pos_regions, neg_regions, pos_examples, neg_examples, idx = dataset1[k].next()
        # 一个next返回的是一个视频序列中随机采样的结果, regions是图像(batch*3*W*H)
        img_list2 = seq['images']
        gt2 = seq['gt']
        img_dir2 = os.path.join(img_home, seqname)
        dataset2[k] = RegionDataset1(img_dir2, img_list2, gt2, pos_regions, neg_regions,
                                     pos_examples, neg_examples, idx, opts)

    # exit()

    ## Init model ##
    model = MDNet(opts['stage'], opts['init_model_path'], K1)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    criterion = BinaryLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts)

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print("==== Start Cycle %d ====" % (i))
        k_list = np.random.permutation(K1)
        prec = np.zeros(K1)
        for j,k in enumerate(k_list):
            tic = time.time()
            pos_regions1, neg_regions1,pos_examples1,neg_examples1,idx1,pos_regions2, neg_regions2,pos_examples2,neg_examples2,idx2 = dataset2[k].next1()
            
            pos_regions1 = torch.tensor(pos_regions1)
            neg_regions1 = torch.tensor(neg_regions1)
            pos_regions2 = torch.tensor(pos_regions2)
            neg_regions2 = torch.tensor(neg_regions2)

        
            if opts['use_gpu']:
                pos_regions1 = pos_regions1.cuda()
                neg_regions1 = neg_regions1.cuda()
                pos_regions2 = pos_regions2.cuda()
                neg_regions2 = neg_regions2.cuda()
        
            pos_score = model(pos_regions1,pos_regions2 ,k)
            neg_score = model(neg_regions1,neg_regions2, k)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])
            optimizer.step()
            
            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print("Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f" % \
                    (i, j, k, loss.item(), prec[k], toc))

        cur_prec = prec.mean()
        print("Mean Precision: %.3f" % (cur_prec))
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {
                    'dim_weight1': model.dim_weight1,
                    'dim_weight2': model.dim_weight2,
                    'dim_weight3': model.dim_weight3,

                    'shared_layers': model.layers.state_dict(),

                    'layers_small1': model.layers_small1.state_dict(),
                    'layers_small2': model.layers_small2.state_dict(),
                    'layers_small3': model.layers_small3.state_dict(),

                    'RGB_para1_3x3': model.RGB_para1_3x3.state_dict(),
                    'RGB_para2_1x1': model.RGB_para2_1x1.state_dict(),
                    'RGB_para3_1x1': model.RGB_para3_1x1.state_dict(),

                    'RGB_VGG1': model.RGB_VGG1.state_dict(),
                    'RGB_VGG2': model.RGB_VGG2.state_dict(),
                    'RGB_VGG3': model.RGB_VGG3.state_dict(),

                    'T_para1_3x3': model.T_para1_3x3.state_dict(),
                    'T_para2_1x1': model.T_para2_1x1.state_dict(),
                    'T_para3_1x1': model.T_para3_1x1.state_dict(),

                    'T_VGG1': model.T_VGG1.state_dict(),
                    'T_VGG2': model.T_VGG2.state_dict(),
                    'T_VGG3': model.T_VGG3.state_dict(),
                    }


            print("Save model to %s" % opts['model_path'])
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-w', '--weight', type=str, default='weights/imagenet-vgg-m.mat',
                        help='Load weight')
    parser.add_argument('-o', '--output', type=str, default='logs/GA.pth',
                        help='Output weight')
    parser.add_argument('-s', '--stage', type=str, default='GA',
                        help='Train stage. Select within \'GA\', \'MA\', \'IA\'')
    parser.add_argument('-I', '--I_Data', type=str, default='rgbt234_I.pkl',
                        help='I_Data. Select within \'rgbt234_I.pkl\', \'rgbt234_I2.pkl\'')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--seed', type=int, default=1,
                        help='Manual seed')

    args = parser.parse_args()
    assert args.stage=='GA' or args.stage=='MA' or args.stage=='IA', 'Please select stage within \'GA\', \'MA\', \'IA\''
    if args.stage == 'GA':
        train_opt = optsGA
    elif args.stage == 'MA':
        train_opt = optsMA
    elif args.stage == 'IA':
        train_opt = optsIA

    data_path2 = 'DATA/' + args.I_Data
    train_opt['init_model_path'] = args.weight
    train_opt['model_path'] = args.output
    train_opt['w_decay'] = args.weight_decay
    np.random.seed(args.seed)
    torch.manual_seed(args.seed * 2)
    torch.cuda.manual_seed(args.seed * 3)
    train_mdnet(train_opt)

    torch.cuda.empty_cache()