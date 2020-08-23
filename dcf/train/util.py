import numpy as np
import cv2

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[1] + 1) - np.floor(float(sz[1]) / 2),
                       np.arange(1, sz[0] + 1) - np.floor(float(sz[0]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    # g = np.roll(g, -1, axis=0)
    g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    return g

def crop_chw(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return np.transpose(crop, (2, 0, 1))

def cwh2corner(bbox):
    if type(bbox) == np.ndarray:
        out = np.zeros(bbox.shape)
        out[0:2] = bbox[0:2] - bbox[2:4] / 2
        out[2:4] = bbox[0:2] + bbox[2:4] / 2
    else:
        out = [0, 0, 0, 0]
        out[0] = bbox[0] - bbox[2] / 2
        out[1] = bbox[1] - bbox[3] / 2
        out[2] = bbox[0] + bbox[2] / 2
        out[3] = bbox[1] + bbox[3] / 2
    return out