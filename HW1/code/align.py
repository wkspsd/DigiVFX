import os
import os.path as osp
import numpy as np
from PIL import Image


def ImageShrink(image):
    small_image = image.resize((image.size[0] // 2, image.size[1] // 2))
    return small_image


def BitmapXOR(bm1, bm2):
    result = np.bitwise_xor(bm1, bm2)
    return result


def BitmapAND(bm1, bm2):
    result = np.bitwise_and(bm1, bm2)
    return result


def BitmapTotal(bm):
    return np.sum(bm)


def BitmapShift(row, col, bm, xo, yo):
    shift_bm = np.roll(bm, xo, axis = 0)
    shift_bm = np.roll(shift_bm, yo, axis = 1)
    
    if xo > 0:
        shift_bm[0:xo,] = False
    elif xo < 0:
        shift_bm[row + xo:row,] = False
    
    if yo > 0:
        shift_bm[:, 0:yo] = False
    elif yo < 0:
        shift_bm[:,col + yo:col] = False

    return shift_bm


def ComputeBitmaps(image):
    image_np = np.array(image)
    upper = np.percentile(image_np, 54)
    lower = np.percentile(image_np, 46)
    mid = np.percentile(image_np, 50)
    col, row = image.size
    tb = image_np > mid
    eb = (image_np > upper) | (image_np < lower)
    return (tb, eb)


def GetExpShift(image1, image2, shift_bits, shift_ret):
    cur_shift = [0, 0]

    if shift_bits > 0:
        sml_img1 = ImageShrink(image1)
        sml_img2 = ImageShrink(image2)
        GetExpShift(sml_img1, sml_img2, shift_bits - 1, cur_shift)
        cur_shift[0] *= 2
        cur_shift[1] *= 2
    else:
        cur_shift[0] = 0
        cur_shift[1] = 0

    tb_1, eb_1 = ComputeBitmaps(image1)
    tb_2, eb_2 = ComputeBitmaps(image2)
    min_err = image1.size[0] * image1.size[1]

    for i in range(-1, 2):
        for j in range(-1, 2):
            x_s = cur_shift[0] + i
            y_s = cur_shift[1] + j
            shifted_tb2 = BitmapShift(image1.size[1], image1.size[0], tb_2, x_s, y_s)
            shifted_eb2 = BitmapShift(image1.size[1], image1.size[0], eb_2, x_s, y_s)
            diff_b = BitmapXOR(tb_1, shifted_tb2)
            diff_b = BitmapAND(diff_b, eb_1) 
            diff_b = BitmapAND(diff_b, shifted_eb2)
            err = BitmapTotal(diff_b)
       
            if err < min_err:
                shift_ret[0] = x_s
                shift_ret[1] = y_s
                min_err = err
       

images =  []
dir = '../data/images'

for filename in np.sort(os.listdir(dir)):
    if osp.splitext(filename)[1] in ['.png', '.jpg']:
        print(filename)
        im = Image.open(osp.join(dir, filename))
        images += [im]

middle_idx = len(images) // 2
base_image = images[middle_idx].convert("L")
images[middle_idx].save("../data/shifted/{0:05d}.jpg".format(middle_idx))
for i in range(len(images)):
    if i != middle_idx:
        target = images[i].convert("L")
        shift_ret = [0, 0]
        GetExpShift(base_image, target, 6, shift_ret)
        shift_image = images[i].transform(images[i].size, Image.Transform.AFFINE, data=(1,0,-shift_ret[1],0,1,-shift_ret[0]))
        shift_image.save("../data/shifted/{0:05d}.jpg".format(i))
