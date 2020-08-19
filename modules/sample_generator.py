import numpy as np
from PIL import Image

from modules.utils import *

def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):
    if bbox[0] < 0.0:
        bbox[0] = 0.0
    if bbox[1] < 0.0:
        bbox[1] = 0.0
    if bbox[0]+bbox[2] > generator.img_size[0]:
        bbox[2] = generator.img_size[0] - bbox[0]
    if bbox[1]+bbox[3] > generator.img_size[1]:
        bbox[3] = generator.img_size[1] - bbox[1]
        
    if overlap_range is None and scale_range is None:
        return generator(bbox, n)
    
    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 16:
            samples_ = generator(bbox, remain*factor)
#            print("hidden")
#print(samples_.shape)
#            print(bbox)
#            print(samples_)
            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                
                r = overlap_ratio(samples_, bbox)
#                print(r)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:,2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])
#                print(idx)
            samples_ = samples_[idx,:]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor*2
#        print('out')
#        print(samples.shape)
#        if samples.shape[0] == 0:
#            ind = []

#            for indx in range (0, n):
#                ind = ind + [np.argmax(r)]
#            print(ind)
        return samples


class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size) # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    def __call__(self, bb, n):
        #
        # bb: target bbox (min_x,min_y,w,h)
        bb = np.array(bb, dtype='float32')
#        if bb[0] < 0.0:
#            bb[0] = 0.0
#        if bb[1] < 0.0:
#            bb[1] = 0.0
#        if bb[0]+bb[2] > self.img_size[0]:
#            bb[2] = self.img_size[0] - bb[0]
#        if bb[1]+bb[3] > self.img_size[1]:
#            bb[3] = self.img_size[1] - bb[1]
#        print(bb)
        # (center_x, center_y, w, h)
        sample = np.array([bb[0]+bb[2]/2, bb[1]+bb[3]/2, bb[2], bb[3]], dtype='float32')
        samples = np.tile(sample[None,:],(n,1))
#        print(samples)
        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n,1)*2-1
            samples[:,2:] *= self.aspect_f ** np.concatenate([ratio, -ratio],axis=1)

        # sample generation
        if self.type=='gaussian':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

        # 在自身bbox的wh范围内最多移动一个身子
        elif self.type=='uniform':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        # 根据n的数量对xy中心画网格随机采样，打乱后取前nge
        elif self.type=='whole':
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
            xy = np.random.permutation(xy)[:n]

            samples[:,:2] = bb[2:]/2 + xy * (self.img_size-bb[2:]/2-1)
            #samples[:,:2] = bb[2:]/2 + np.random.rand(n,2) * (self.img_size-bb[2:]/2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        # adjust bbox range
        samples[:,2:] = np.clip(samples[:,2:], 3, self.img_size-3)
        if self.valid:
            samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, self.img_size-samples[:,2:]/2-1)
        else:
            samples[:,:2] = np.clip(samples[:,:2], 0, self.img_size)

        # (min_x, min_y, w, h)
        samples[:,:2] -= samples[:,2:]/2

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f
    
    def get_trans_f(self):
        return self.trans_f

if __name__ == '__main__':
    test = SampleGenerator('uniform', (640, 512), trans_f=1, aspect_f=None)
    a = test([100, 150, 30, 20], 10)
    print(a)

