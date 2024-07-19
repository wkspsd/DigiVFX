import cv2
import numpy as np
import math
from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist


# Harris Corner Detection
def Corner(gray, ksize = 9, S = 3, k = 0.04):
    K = (ksize, ksize)
    gray_blur = cv2.GaussianBlur(gray, K, S)
    Iy, Ix = np.gradient(gray_blur)
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix * Iy
    Sxx = cv2.GaussianBlur(Ixx, K, S)
    Syy = cv2.GaussianBlur(Iyy, K, S)
    Sxy = cv2.GaussianBlur(Ixy, K, S)
    detM = (Sxx * Syy) - (Sxy**2)
    traceM = Sxx + Syy
    R = detM - k * (traceM**2)

    return R, Ix, Iy

# Find local maximum 
def local_max_R(R, threshold = 0.01):
    kernel = np.ones((3, 3), dtype = np.float32) * -1
    kernel[1, 1] = 8
    localMax = R > np.max(R) * threshold
    filtered_image = np.abs(signal.convolve2d(localMax, kernel, mode = 'same'))
    localMax = (filtered_image == np.max(filtered_image))
    feature_points = np.argwhere(localMax)
    
    return feature_points


def get_patches(rotated_img, pos):
    up = pos[0] - 20 if (pos[0] - 20) >= 0 else 0
    down = pos[0] + 20 if (pos[0] + 20) < rotated_img.shape[0] else rotated_img.shape[0]
    left = pos[1] - 20 if (pos[1] - 20) >= 0 else 0
    right = pos[1] + 20 if (pos[1] + 20) < rotated_img.shape[1] else rotated_img.shape[1]

    return rotated_img[up : down, left : right]

# Calculate MSOP descriptors
def MSOP_descriptors(src_img, feature_pos, Ix, Iy):
    len_fp = len(feature_pos[0])
    desc_left_list = []
    desc_right_list = []
    for i in range(len_fp):
        pos = (int(feature_pos[0][i]), int(feature_pos[1][i]))
        rotated_shape = cv2.getRotationMatrix2D(pos, math.atan2(Iy[pos], Ix[pos]), scale = 1.0)
        rotated_img = cv2.warpAffine(src_img, rotated_shape, (src_img.shape[1], src_img.shape[0]))
        patch_40 = get_patches(rotated_img, pos)
        gauss_patch_40 = gaussian_filter(patch_40, sigma = 1) 
        patch_8 = cv2.resize(gauss_patch_40, (8, 8))
        normal_patch_8 = (patch_8 - np.mean(patch_8)) / (patch_8 + 1e-8)
        desc = {"coordinate": pos, "orientation": (Iy[pos], Ix[pos]), "patch": normal_patch_8.flatten().tolist()}
        if pos[1] < src_img.shape[1] // 2:
            desc_left_list.append(desc)
        else:
            desc_right_list.append(desc)

    return desc_left_list, desc_right_list


def match_feature(desc_right, desc_left, threshold = 0.8):
    img1_all_right_patches = []
    img2_all_left_patches = []
    for desc in desc_right:
        img1_all_right_patches.append(desc["patch"])

    for desc in desc_left:
        img2_all_left_patches.append(desc["patch"])

    all_combination_dist = cdist(img1_all_right_patches, img2_all_left_patches)
    num_desc_right = len(desc_right)
    num_desc_left = len(desc_left)
    all_fs_matches = []

    for i in range(num_desc_right):
        first_closest_index = 0
        second_closest_index = 0
        first_closest_dist = float("inf")
        second_closest_dist = float("inf")

        for j in range(num_desc_left):
            if all_combination_dist[i, j] < second_closest_dist and all_combination_dist[i, j] < first_closest_dist:
                second_closest_dist = first_closest_dist
                second_closest_index = first_closest_index
                first_closest_dist = all_combination_dist[i, j]
                first_closest_index = (i, j)

            elif all_combination_dist[i, j] < second_closest_dist and all_combination_dist[i, j] >= first_closest_dist:
                second_closest_dist = all_combination_dist[i, j]
                second_closest_index = (i, j)
                
        all_fs_matches.append((first_closest_index, second_closest_index))

    matched_indexes = []
    for fs in all_fs_matches:
        first_closest = all_combination_dist[fs[0][0], fs[0][1]]
        second_closest = all_combination_dist[fs[1][0], fs[1][1]]
        
        if second_closest == 0:
            continue

        if first_closest / second_closest < threshold:
            matched_indexes.append(fs[0])

    return matched_indexes


def warp_feature(h, w, desc_list, focal_length = 1000):
    y_o = h // 2
    x_o = w // 2
    numm_descs = len(desc_list)

    for i in range(numm_descs):
        x_p = math.atan(((desc_list[i]["coordinate"][1] - x_o) / focal_length )) * focal_length  + x_o
        y_p =  focal_length  * (desc_list[i]["coordinate"][0] - y_o) /\
        math.sqrt((desc_list[i]["coordinate"][1] - x_o)**2 + focal_length **2) + y_o
        desc_list[i]["coordinate"] = (round(y_p), round(x_p))

    return
