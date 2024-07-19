import cv2
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from warping import cylinder_warping
from feature import Corner, local_max_R, get_patches, MSOP_descriptors, match_feature, warp_feature


ransac_list = []
def RANSAC(desc_right, desc_left, matched_index):
    max_inlier = 0
    shift_xy = []

    for i in range(100000):
        num_chosen = 1
        chosen = random.sample(matched_index, k = num_chosen)
        curr_shift = [desc_right[chosen[0][0]]["coordinate"][0] - desc_left[chosen[0][1]]["coordinate"][0], \
                      desc_right[chosen[0][0]]["coordinate"][1] - desc_left[chosen[0][1]]["coordinate"][1]]
        for j in range(1, num_chosen):
            curr_shift[0] += desc_right[chosen[j][0]]["coordinate"][0] - desc_left[chosen[j][1]]["coordinate"][0]
            curr_shift[1] += desc_right[chosen[j][0]]["coordinate"][1] - desc_left[chosen[j][1]]["coordinate"][1]
        curr_shift[0] /= num_chosen
        curr_shift[1] /= num_chosen
        curr_inlier = 0
        choosed = []

        for g, h in matched_index:
            if [g, h] not in chosen:
                other_shift = [desc_right[g]["coordinate"][0] - desc_left[h]["coordinate"][0], \
                               desc_right[g]["coordinate"][1] - desc_left[h]["coordinate"][1]]
                if abs(curr_shift[0] - other_shift[0]) < 4 and abs(curr_shift[1] - other_shift[1]) < 4:
                    curr_inlier += 1
                    choosed.append([g, h])

        if max_inlier < curr_inlier:
            max_inlier = curr_inlier
            for g, h in choosed:
                curr_shift[0] += desc_right[g]["coordinate"][0] - desc_left[h]["coordinate"][0]
                curr_shift[1] += desc_right[g]["coordinate"][1] - desc_left[h]["coordinate"][1]
            curr_shift[0] /= (max_inlier + 1)
            curr_shift[1] /= (max_inlier + 1)
            shift_xy = curr_shift
            global ransac_list
            ransac_list = choosed

    shift_xy[0] = round(shift_xy[0])
    shift_xy[1] = round(shift_xy[1])

    return shift_xy


def find_local_max_R(R, threshold = 0.01):
    localMax = np.zeros(R.shape, dtype = np.uint8)
    localMax[R > np.max(R) * threshold] = 1
    for y in range(3):
        for x in range(3):
            if x == 1 and y == 1:
                continue
            kernels = np.zeros((3, 3))
            kernels[1, 1] = 1
            kernels[y, x] = -1
            result = cv2.filter2D(R, -1, kernels)
            localMax[result < 0] = 0 

    feature_points = np.where(localMax > 0)
    
    return feature_points


def init_stitching_space(images, total_shifts):
    h, w, c = images[0].shape
    if total_shifts[0] > 0:
        h += total_shifts[0] 

    w += total_shifts[1]

    return np.zeros((h, w, c), dtype = np.uint8)

# Image stitching with linear blending
def image_stitching(stitching_space, images, all_shifts):
    cumulated_shifts = [0, 0]
    cumulated_h, cumulated_w, temp_c = images[0].shape
    num_images = len(images)
    stitching_space[:images[0].shape[0], :images[0].shape[1], :] = images[0]

    for i in range(num_images - 1):
        cumulated_shifts[0] = cumulated_shifts[0] + all_shifts[i][0]
        cumulated_shifts[1] = cumulated_shifts[1] + all_shifts[i][1]
        overlapped_col = cumulated_w - cumulated_shifts[1]
        left_part = overlapped_col - 1
        right_part = 1 
        h, w, c = images[i + 1].shape

        if cumulated_shifts[0] >= 0:
            start_row = cumulated_shifts[0]  
            end_row = h + start_row
        else:
            start_row = 0
            end_row = h + cumulated_shifts[0]

        if cumulated_shifts[0] >= 0:
            im_start_row = 0
        else:
            im_start_row = -cumulated_shifts[0]
        
        for j in range(cumulated_shifts[1], cumulated_w):
            if (stitching_space[(start_row + (end_row)) // 2, (j - 1), 1] == 0):
                stitching_space[start_row:end_row, (j - 1), :] = images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :]
            elif (images[i + 1][h // 2, (j - cumulated_shifts[1]), 1] == 0):
                pass
            else:
                stitching_space[start_row:end_row, (j - 1), :] = \
                stitching_space[start_row:end_row, (j - 1), :]  / overlapped_col * left_part + \
                images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :] / overlapped_col * right_part
            left_part -= 1
            right_part += 1

        stitching_space[start_row:end_row, cumulated_w - 1:w + cumulated_shifts[1], :] = \
        images[i + 1][im_start_row:, (cumulated_w - cumulated_shifts[1] - 1):, :]
        cumulated_h += all_shifts[i][0]
        cumulated_w += all_shifts[i][1]

    return stitching_space.astype(np.uint8)

# Image stitching without blending
def image_stitching1(stitching_space, images, all_shifts):
    cumulated_shifts = [0, 0]
    cumulated_h, cumulated_w, temp_c = images[0].shape
    num_images = len(images)
    stitching_space[:images[0].shape[0], :images[0].shape[1], :] = images[0]

    for i in range(num_images - 1):
        cumulated_shifts[0] = cumulated_shifts[0] + all_shifts[i][0]
        cumulated_shifts[1] = cumulated_shifts[1] + all_shifts[i][1]
        overlapped_col = cumulated_w - cumulated_shifts[1]
        left_part = overlapped_col - 1
        right_part = 1 
        h, w, c = images[i + 1].shape
        if cumulated_shifts[0] >= 0:
            start_row = cumulated_shifts[0]  
            end_row = h + start_row
        else:
            start_row = 0
            end_row = h + cumulated_shifts[0]

        if cumulated_shifts[0] >= 0:
            im_start_row = 0
        else:
            im_start_row = -cumulated_shifts[0]
        
        for j in range(cumulated_shifts[1], cumulated_w):
            if (j > (cumulated_shifts[1] + cumulated_w) / 2):
                stitching_space[start_row:end_row, (j - 1), :] = images[i + 1][im_start_row:, (j - cumulated_shifts[1]), :]
            left_part -= 1
            right_part += 1
        stitching_space[start_row:end_row, cumulated_w - 1:w + cumulated_shifts[1], :] = \
        images[i + 1][im_start_row:, (cumulated_w - cumulated_shifts[1] - 1):, :]
        cumulated_h += all_shifts[i][0]
        cumulated_w += all_shifts[i][1]

    return stitching_space.astype(np.uint8)


def bundle_adjust(img):
    h, w, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for x in range(0, w):
        find = False
        for y in range(0, h):
            if img_gray[y, x] > 0:
                upper_left = [x, y]
                find = True
                break
        if find:
            break

    for x in range(0, w):
        find = False
        for y in range(h - 1, -1, -1):
            if img_gray[y, x] > 0:
                bottom_left = [x, y]
                find = True
                break
        if find:
            break

    for x in range(w - 1, -1, -1):
        find = False
        for y in range(0, h):
            if img_gray[y, x] > 0:
                upper_right = [x, y]
                find = True
                break
        if find:
            break

    for x in range(w - 1, -1, -1):
        find = False
        for y in range(h - 1, -1, -1):
            if img_gray[y, x] > 0:
                bottom_right = [x, y]
                find = True
                break
        if find:
            break

    corner = np.float32([upper_left, upper_right, bottom_left, bottom_right])
    img_corner = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    M = cv2.getPerspectiveTransform(corner, img_corner)
    pano_adjust = cv2.warpPerspective(img, M, (w, h))

    return pano_adjust


if __name__ == '__main__':
    filenames = []
    bgrs = []
    focal_length = [1701.34, 1702.8, 1704.8]
    dir = '../data/images'
    print('Load Images...')
    for filename in np.sort(os.listdir(dir)):
        if osp.splitext(filename)[1] in ['.png', '.jpg']:
            print(filename)
            filenames.append(filename[:-4])
            im = cv2.imread(osp.join(dir, filename))
            bgrs.append(im) 

    n = 3
    print('Cylinder Warping...')
    warp = [cylinder_warping(bgrs[i], filenames[i], focal_length = focal_length[i]) for i in range(n)]

    grays = []

    for i in range(n):
        grays.append(cv2.cvtColor(bgrs[i], cv2.COLOR_BGR2GRAY).astype(np.float32))

    R, Ix, Iy = [], [], []
    print('Feature Detection & Feature Matching...')
    for i in range(n):
        r, ix, iy = Corner(grays[i])
        R.append(r)
        Ix.append(ix)
        Iy.append(iy)

    fpts = []
    for i in range(n):
        fpts.append(find_local_max_R(R[i]))

    for i in range(n):    
        img_fpts = np.copy(bgrs[i])
        for j in range(len(fpts[i][0])):
            cv2.circle(img_fpts, (fpts[i][1][j], fpts[i][0][j]), radius = 1, color = [0, 0, 255], thickness = 1, lineType = 1)
        cv2.imwrite(f'../data/results/{filenames[i]}_fpts.png', img_fpts)

    desc_left_list, desc_right_list = [], []
    for i in range(n):
        left, right = MSOP_descriptors(bgrs[i], fpts[i], Ix[i], Iy[i])
        desc_left_list.append(left)
        desc_right_list.append(right)

    test_height = bgrs[0].shape[0]
    test_width = bgrs[0].shape[1]

    for i in range(n):
        warp_feature(test_height, test_width, desc_left_list[i], focal_length[i])
        warp_feature(test_height, test_width, desc_right_list[i], focal_length[i])

    matches = []
    all_shifts = []
    for i in range(n - 1):
        matches.append(match_feature(desc_right_list[i], desc_left_list[i + 1], threshold = 1))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
        ax1.imshow(cv2.cvtColor(warp[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
        ax2.imshow(cv2.cvtColor(warp[i + 1].astype(np.uint8), cv2.COLOR_BGR2RGB))
        all_shifts.append(RANSAC(desc_right_list[i], desc_left_list[i + 1], matches[i]))

        for match in ransac_list:
            con = ConnectionPatch(xyA = (desc_left_list[i + 1][match[1]]["coordinate"][1], desc_left_list[i + 1][match[1]]["coordinate"][0]), \
                                xyB = (desc_right_list[i][match[0]]["coordinate"][1], desc_right_list[i][match[0]]["coordinate"][0]), \
                                coordsA = "data", coordsB = "data", \
                                axesA = ax2, axesB = ax1, color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
            fig.add_artist(con)
        plt.savefig(f'../data/results/{filenames[i]}_{filenames[i+1]}_lines.png')

    max_shifts = [0, 0]
    for i, j in all_shifts:
        max_shifts[0] += max(max_shifts[0], i)
        max_shifts[1] += max(max_shifts[1], j)

    print('Image Matching & Image Blending...')
    stitching_space = init_stitching_space(bgrs, max_shifts)
    stitched_blending = image_stitching(stitching_space, warp, all_shifts)
    stitched_no_blending = image_stitching1(stitching_space, warp, all_shifts)
    cv2.imwrite(f'../data/results/linear_blending.png', stitched_blending)
    cv2.imwrite(f'../data/results/no_blending.png', stitched_no_blending)
    print('Bundle Adjustment...')
    bundle = bundle_adjust(stitched_no_blending)
    cv2.imwrite(f'../data/results/result.png', bundle)



