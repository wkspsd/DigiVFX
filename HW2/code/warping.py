import cv2
import numpy as np


def cylinder_warping(image, file_name, focal_length = 1000):
    h, w, c = image.shape
    s = focal_length
    res = np.zeros([h, w, 3])
    y_o = np.floor(h / 2)
    x_o = np.floor(w / 2)
    y, x = np.indices((h, w))
    y_p = y - y_o
    x_p = x - x_o
    x = focal_length * np.tan(x_p / s)
    y = np.sqrt(x**2 + focal_length**2) / s * y_p
    y += y_o
    x += x_o
    x_f = np.clip(np.floor(x).astype(int), 0, w - 1)
    y_f = np.clip(np.floor(y).astype(int), 0, h - 1)
    x_c = np.clip(np.ceil(x).astype(int), 0, w - 1)
    y_c = np.clip(np.ceil(y).astype(int), 0, h - 1)
    idx = np.ones([h,w])
    idx[np.floor(x) < 0] = 0; idx[np.floor(y) < 0] = 0; idx[np.ceil(x) > w - 1] = 0; idx[np.ceil(y) > h - 1] = 0
    a = x - x_f
    b = y - y_f

    for i in range(c):
        res[..., i] = (1 - a) * (1 - b) * image[y_f, x_f, i] + a * (1 - b) * image[y_f, x_c, i] + a * b * image[y_c, x_c, i] + (1 - a) * b * image[y_c, x_f, i]

    res[idx == 0] = [0, 0, 0]
    cv2.imwrite(f'../data/results/{file_name}_warp.png', res)
    
    return res
