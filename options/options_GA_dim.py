from collections import OrderedDict

opts = OrderedDict()
opts['stage'] = 'GA'
opts['use_gpu'] = True

opts['init_model_path'] = 'weights/imagenet-vgg-m.mat'

opts['model_path'] = 'logs/GA.pth'

# opts['batch_frames'] = 2
# opts['batch_pos'] = 8
# opts['batch_neg'] = 24
# opts['n_cycles'] = 1
opts['batch_frames'] = 8
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['n_cycles'] = 150

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['fc', 'R', 'T', 'conv', 'dim']
opts['lr_mult'] = {'fc': 10, 'R': 1, 'T': 1, 'conv': 1, 'dim': 100000}

