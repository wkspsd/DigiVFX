import os
import os.path as osp
import cv2
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def solve_debevec(z, lng, N, P):
    print('solving debevec')
    z = np.transpose(z)
    A = np.zeros((N * P + 255, 256 + N))
    B = np.zeros((N * P + 255))
    lamdas = 10
    
    for i in range(N):
        for j in range(P):
            w_ij = W[z[i][j]]
            A[P * i + j, z[i][j]] = w_ij
            A[P * i + j, i + 256] = -w_ij
            B[P * i + j] = w_ij * lng[j]

    A[N * P, 127] = 1

    for i in range(254):
        A[N * P + 1 + i, i:i + 3] = np.array([1, -2, 1]) * W[i] * (lamdas**2)
    x = np.linalg.lstsq(A, B, rcond = None)[0].ravel()
    return x[:256] 


def get_radiance_map(images, lng_bgr):
    Ln_radiance_bgr = np.zeros([height, width, 3]).astype(np.float32)

    P = len(images)
    plt.clf()
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    bgr_titles = ['blue', 'green', 'red']
    rads = np.empty((height, width, 3)).astype(np.float32)
    for c in range(3):

        W_sum = np.zeros([height, width], dtype = np.float32)
        Ln_rad_sum = np.zeros([height, width], dtype = np.float32)

        for p in range(P):
                        
            im_1D = images[p][:, :, c].flatten()
            Ln_rad = (lng_bgr[c][im_1D] - lng[p]).reshape(height, width)
            weights = W[im_1D].reshape(height, width)
            w_Ln_rad = Ln_rad * weights
            Ln_rad_sum += w_Ln_rad
            W_sum += weights

        weighted_Ln_radiance = Ln_rad_sum / W_sum
        rads[:,:,c] = (np.exp(weighted_Ln_radiance))
        plt.figure()
        plt.imshow(weighted_Ln_radiance, cmap = plt.cm.jet )
        plt.colorbar()
        plt.savefig('../data/hdr/radiance_map_{}.png'.format(bgr_titles[c]))
        plt.close()
    cv2.imwrite('../data/hdr/result.hdr', rads)
    return rads
 

def display_response_curve(LnG_bgr):
    bgr_titles = ['blue', 'green', 'red']
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))
    for i in range(3):
        ax = axes[i]
        ax.plot(LnG_bgr[i], np.arange(256), c = bgr_titles[i])
        ax.set_title(bgr_titles[i])
        ax.set_xlabel('E')
        ax.set_ylabel('Z')
        ax.grid(linestyle = ':', linewidth = 1)
    fig.savefig('../data/hdr/res_curve.png', bbox_inches = 'tight', dpi = 256)


def global_operator(image, d = 1e-6, key_value = 0.36):
    print('2002_global_tonemapping')
    Lum = 0.06 * image[:, :, 0] + 0.67 * image[:, :, 1] + 0.27 * image[:, :, 2]
    L_w_ave = math.exp(np.mean(np.log(d + Lum))) 
    scaled_Lum = key_value * Lum / L_w_ave
    L_white = np.max(Lum)
    L_d = (scaled_Lum * (1 + scaled_Lum / (L_white ** 2))) / (1 + scaled_Lum)
    for i in range(3):
        image[:, :, i] = L_d * image[:, :, i] / Lum
    image = np.clip(image * 255, 0, 255)
    return image.astype(np.uint8)


def local_operator(image, d = 1e-6, key_value = 0.36, phi = 8.0, epsilon = 0.05):
    print('2002_local_tonemapping')
    Lum = 0.06 * image[:, :, 0] + 0.67 * image[:, :, 1] + 0.27 * image[:, :, 2]
    L_w_ave = math.exp(np.mean(np.log(d + Lum))) 
    scaled_Lum = key_value * Lum / L_w_ave
    L_blur = []
    for s in range(1, 18, 2):
        L_blur.append(cv2.GaussianBlur(scaled_Lum, (s, s), 0))
    V = []
    length_L_blur = len(L_blur)
    for i in range(length_L_blur - 1):
        V.append((L_blur[i] - L_blur[i + 1]) / (2**phi * key_value / (i * 2 + 1)**2 + L_blur[i]))

    L_d = np.zeros(Lum.shape)
    counter = 0
    for x in range(Lum.shape[0]):
        for y in range(Lum.shape[1]):
            for index in range(8):
                if abs(V[index][x, y]) < epsilon:
                    counter += 1
                    L_d[x, y] = scaled_Lum[x, y] / (1 + L_blur[index][x, y])
                    break
    for i in range(3):
        image[:, :, i] = L_d * image[:, :, i] / Lum
    image = np.clip(image * 255, 0, 255)
    return image.astype(np.uint8)


def global_tone_map(image, f, m, a, c):
    print('2005_global_tonemapping')
    Cav = []
    for i in range(3):
        average = np.mean(image[:, :, i])
        Cav.append(average)
    Lum = 0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.2125 * image[:, :, 2]
    Lav = np.mean(Lum)
    f = math.exp(-f)
    Llav = np.mean(np.log(1e-6 + Lum))
    Lmax = np.max(Lum)
    Lmin = np.min(Lum)
    m = m if m > 0 else (0.3 + 0.7 * math.pow((math.log(Lmax) - Llav) / (math.log(Lmax) - math.log(Lmin)), 1.4))
    for i in range(3):
        I_l = c * image[:,:,i] + (1 - c) * Lum
        I_g = c * Cav[i] + (1 - c) * Lav
        I_a = a * I_l + (1 - a) * I_g
        image[:, :, i] = image[:, :, i] / (image[:, :, i] + np.power(f * I_a, m))
    image = np.clip(image * 255, 0, 255)
    return image.astype(np.uint8)


images =  []
dir = '../data/shifted'

for filename in np.sort(os.listdir(dir)):
    if osp.splitext(filename)[1] in ['.png', '.jpg']:
        print(filename)
        im = cv2.imread(osp.join(dir, filename))
        images += [im]

P = len(images)
(height, width, channel) = images[0].shape
shutter_times = np.array([1/8000, 1/6000, 1/4000, 1/3000, 1/2000, 1/1500, 1/1000, 1/750, 1/500, 1/320, 1/200, 1/125, 1/80, 1/60, 1/40, 1/25, 1/15, 1/8])
lng = np.log(shutter_times).astype(np.float32)
N = 200
random.seed(7414)
indices = np.array(random.sample(range(height * width), N))

print('P =', P)
print('h', height, 'w', width)
print('N', N)

col = indices % width
row = indices % height
Z_bgr = [[images[p][row, col, i] for p in range(P)] for i in range(3)]

weight_type = 1

W = np.concatenate((np.arange(1, 129), np.arange(1, 129)[::-1]), axis = 0)

LnG_bgr_debevec = [solve_debevec(Z, lng, N, P) for Z in Z_bgr]

radiance = get_radiance_map(images, LnG_bgr_debevec)

display_response_curve(LnG_bgr_debevec)

image = cv2.imread("../data/hdr/result.hdr", flags = cv2.IMREAD_ANYDEPTH)
new_image = global_operator(image)
new_image2 = local_operator(image, key_value=0.72)
cv2.imwrite("../data/tone-mapped/global_2002.jpg", new_image)
cv2.imwrite("../data/tone-mapped/local_2002.jpg", new_image2)

image = cv2.imread("../data/hdr/result.hdr", flags = cv2.IMREAD_ANYDEPTH)
image = global_tone_map(image, -4, 0, 0, 0)
cv2.imwrite("../data/tone-mapped/global_2005.jpg", image)