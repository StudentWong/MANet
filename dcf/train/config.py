from dcf.train.util import *
import torch

class TrackerConfig(object):
    # These are the default hyper-params for DCFNet
    # OTB2013 / AUC(0.665)
    feature_path = '/home/studentw/disk3/tracker/MANet/weights/imagenet-vgg-m.mat'
    file = '/home/studentw/disk3/tracker/RGB_T234/rgbt234.txt'
    root = '/home/studentw/disk3/tracker/RGB_T234/'

    crop_sz = 125
    layer = 2

    lr = 0.0001
    w_decay = 0.0005
    momentum = 0.9
    grad_clip = 10
    ft_layers = ['share', 'R', 'T', 'fusion']
    lr_mult = {'share': 1, 'R': 1, 'T': 1, 'fusion': 10}

    lambda0 = 1e-4
    padding = 2
    output_sigma_factor = 0.1
    interp_factor = 0.01
    num_scale = 3
    #num_scale = 5
    scale_step = 1.0275
    #scale_step = 1.3
    scale_factor = scale_step ** (np.arange(num_scale) - num_scale / 2)
    #scale_factor = scale_step ** (np.arange(num_scale) - int(num_scale / 2))
    min_scale_factor = 0.2
    max_scale_factor = 5
    scale_penalty = 0.9925
    scale_penalties = scale_penalty ** (np.abs((np.arange(num_scale) - num_scale / 2)))

    net_input_size = [crop_sz, crop_sz]
    net_average_image = np.array([104, 117, 123]).reshape(-1, 1, 1).astype(np.float32)
    #net_average_image = np.array([0, 0, 0]).reshape(-1, 1, 1).astype(np.float32)
    output_sigma = crop_sz / (1 + padding) * output_sigma_factor
    y = gaussian_shaped_labels(output_sigma, net_input_size)
    #print(y.shape)
    yf = torch.rfft(torch.Tensor(y).view(1, 1, crop_sz, crop_sz).cuda(), signal_ndim=2)
    #print(yf.shape)
    cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()
