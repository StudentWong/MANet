import argparse
import shutil
from os.path import join, isdir, isfile
from os import makedirs

from dcf.train.dataset import VID
from dcf.train.net_train import DCFNet, set_optimizer
from dcf.train.config import TrackerConfig
import torch
from torch.utils.data import dataloader
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import time


parser = argparse.ArgumentParser(description='Training DCFNet in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
parser.add_argument('--train_data', default='rgbt234', type=str, help='train data')
parser.add_argument('--val_data', default='gtot', type=str, help='val data')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
parser.add_argument('--layer', default=2, type=int, help='layer num')

args = parser.parse_args()

print(args)
config = TrackerConfig()
assert args.layer == 2 or args.layer == 3, 'invalid layer'
assert args.train_data=='gtot' or args.train_data=='rgbt234', 'invalid train data'
assert args.val_data=='gtot' or args.val_data=='rgbt234', 'invalid val data'

config.layer = args.layer
config.train_data = args.train_data
config.val_data = args.val_data
if config.train_data == 'rgbt234':
    config.train_file = config.rgbt234_file
    config.train_root = config.rgbt234_root
elif config.train_data == 'gtot':
    config.train_file = config.gtot_file
    config.train_root = config.gtot_root

if config.val_data == 'rgbt234':
    config.val_file = config.rgbt234_file
    config.val_root = config.rgbt234_root
elif config.val_data == 'gtot':
    config.val_file = config.gtot_file
    config.val_root = config.gtot_root


best_loss = 1e6
config.w_decay = args.weight_decay

# def gaussian_shaped_labels(sigma, sz):
#     x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
#     d = x ** 2 + y ** 2
#     g = np.exp(-0.5 / (sigma ** 2) * d)
#     g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
#     g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
#     return g.astype(np.float32)


# class TrackerConfig(object):
#     crop_sz = 125
#     output_sz = 121
#
#     lambda0 = 1e-4
#     padding = 2.0
#     output_sigma_factor = 0.1
#
#     output_sigma = crop_sz / (1 + padding) * output_sigma_factor
#     y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])
#     yf = torch.rfft(torch.Tensor(y).view(1, 1, output_sz, output_sz).cuda(), signal_ndim=2)
#     # cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window



model = DCFNet(config=config)
model.cuda()
gpu_num = torch.cuda.device_count()
print('GPU NUM: {:2d}'.format(gpu_num))
if gpu_num > 1:
    model = torch.nn.DataParallel(model, list(range(gpu_num))).cuda()

criterion = nn.MSELoss(size_average=False).cuda()

# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)

optimizer = set_optimizer(model.feature, config)

# target = torch.Tensor(config.y).cuda().unsqueeze(0).unsqueeze(0).repeat(args.batch_size * gpu_num, 1, 1, 1)  # for training
# optionally resume from a checkpoint
if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

# training data
# crop_base_path = join('dataset', 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
#
# if not isdir(crop_base_path):
#     print('please run gen_training_data.py --output_size {:d} --padding {:.1f}!'.format(args.input_sz, args.padding))
#     exit()

save_path = join(args.save, 'crop_{:d}_{:1.1f}'.format(args.input_sz, args.padding))
if not isdir(save_path):
    makedirs(save_path)


train_dataset = VID(file=config.train_file, root=config.train_root, data=config.train_data, padding=config.padding, fixsize=config.crop_sz)
val_dataset = VID(file=config.val_file, root=config.val_root, data=config.val_data, padding=config.padding, fixsize=config.crop_sz)
# print(len(train_dataset))
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size*gpu_num, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
# print(len(train_loader))
val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size*gpu_num, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)


# def adjust_learning_rate(optimizer, epoch):
#     lr = np.logspace(-2, -5, num=args.epochs)[epoch]
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename=join(save_path, 'checkpoint.pth.tar')):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (tr, ti, sr, si, res) in enumerate(train_loader):
        # print(i)
        # print(tr.shape)
        # measure data loading time
        data_time.update(time.time() - end)

        tr = tr.cuda(non_blocking=True)
        ti = ti.cuda(non_blocking=True)
        sr = sr.cuda(non_blocking=True)
        si = si.cuda(non_blocking=True)
        res = res.cuda(non_blocking=True)

        # compute output
        output = model(tr, ti, sr, si)
        loss = criterion(output, res)/tr.size(0)  # criterion = nn.MSEloss

        # measure accuracy and record loss
        losses.update(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    return losses


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (tr, ti, sr, si, res) in enumerate(val_loader):

            # compute output
            tr = tr.cuda(non_blocking=True)
            ti = ti.cuda(non_blocking=True)
            sr = sr.cuda(non_blocking=True)
            si = si.cuda(non_blocking=True)
            res = res.cuda(non_blocking=True)

            # compute output
            # output = model(template, search)
            # loss = criterion(output, target)/(args.batch_size * gpu_num)
            output = model(tr, ti, sr, si)
            loss = criterion(output, res) / tr.size(0)  # criterion = nn.MSEloss

            # measure accuracy and record loss
            losses.update(loss.item())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses))

        print(' * Loss {loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))

    return losses


for epoch in range(args.start_epoch, args.epochs):
    # adjust_learning_rate(optimizer, epoch)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # print("avg_loss:{:f}".format(loss.avg))
    # evaluate on validation set
    losses = validate(val_loader, model, criterion)
    #
    # # remember best loss and save checkpoint
    is_best = losses.avg < best_loss
    best_loss = min(best_loss, losses.avg)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_loss': best_loss,
        'optimizer': optimizer.state_dict(),
    }, is_best)
