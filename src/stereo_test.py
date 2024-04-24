import torch
from numba import njit, prange
from numba_progress import ProgressBar
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from torchvision.utils import save_image
import cv2
from tqdm import tqdm

def read_image(image_path, if_depthmap=False):
    I = Image.open(image_path)
    if if_depthmap:
        I = I.convert('L')
    #I.show()
    #I.save('./save.png')
    I_array = np.array(I)

    return I_array

def create_stereoimages(original_image, depthmap, divergence, separation=0.0, modes=None,
                        stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp'):

    original_image = np.asarray(original_image)
    balance = (stereo_balance + 1) / 2

    from numba_progress import ProgressBar
    with ProgressBar(total=original_image.shape[0]) as progress:
        left_eye = original_image if balance < 0.001 else \
            apply_stereo_divergence(original_image, depthmap, +1 * divergence * balance, -1 * separation,
                                    stereo_offset_exponent, fill_technique, progress=progress)

        right_eye = original_image if balance > 0.999 else \
            apply_stereo_divergence(original_image, depthmap, -1 * divergence * (1 - balance), separation,
                                    stereo_offset_exponent, fill_technique, progress=progress)

    b, g, r = cv2.split(left_eye)
    left_eye = cv2.merge([r, g, b])
    cv2.imwrite("result/left_eye.jpg", left_eye)
    b, g, r = cv2.split(right_eye)
    right_eye = cv2.merge([r, g, b])
    cv2.imwrite("result/right_eye.jpg", right_eye)

    cv2.imwrite("result/all.jpg",np.hstack([left_eye, right_eye]))


def apply_stereo_divergence(original_image, depth, divergence, separation, stereo_offset_exponent, fill_technique, progress):
    assert original_image.shape[:2] == depth.shape, 'Depthmap and the image must have the same size'
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    divergence_px = (divergence / 100.0) * original_image.shape[1]
    separation_px = (separation / 100.0) * original_image.shape[1]

    return apply_stereo_divergence_polylines(
            original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique, progress_proxy=progress
        )

@njit(parallel=False)  # fastmath=True does not reasonably improve performance
def apply_stereo_divergence_polylines(
        original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float,
        fill_technique: str, progress_proxy):
    # This code treats rows of the image as polylines
    # It generates polylines, morphs them (applies divergence) to them, and then rasterizes them
    EPSILON = 1e-7
    PIXEL_HALF_WIDTH = 0.45 if fill_technique == 'polylines_sharp' else 0.0
    # PERF_COUNTERS = [0, 0, 0]

    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    for row in prange(h):
        progress_proxy.update(1)
        pt = np.zeros((5 + 2 * w, 3), dtype=np.float_)
        pt_end: int = 0
        pt[pt_end] = [-1.0 * w, 0.0, 0.0]
        pt_end += 1
        for col in range(0, w):
            coord_d = (normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px
            coord_x = col + 0.5 + coord_d + separation_px
            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2
        pt[pt_end] = [2.0 * w, 0.0, w - 1]
        pt_end += 1

        # generating the segments of the morphed polyline
        # format: coord_x, coord_d, color_i of the first point, then the same for the second point
        sg_end: int = pt_end - 1
        sg = np.zeros((sg_end, 6), dtype=np.float_)
        for i in range(sg_end):
            sg[i] += np.concatenate((pt[i], pt[i + 1]))

        for i in range(1, sg_end):
            u = i - 1
            while pt[u][0] > pt[u + 1][0] and 0 <= u:
                pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                u -= 1

        # rasterizing
        # at each point in time we keep track of segments that are "active" (or "current")
        csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float_)
        csg_end: int = 0
        sg_pointer: int = 0
        # and index of the point that should be processed next
        pt_i: int = 0
        for col in range(w):  # iterate over regions (that will be rasterized into pixels)
            color = np.full(c, 0.5, dtype=np.float_)  # we start with 0.5 because of how floats are converted to ints
            while pt[pt_i][0] < col:
                pt_i += 1
            pt_i -= 1  # pt_i now points to the dot before the region start
            # Finding segment' parts that contribute color to the region
            while pt[pt_i][0] < col + 1:
                coord_from = max(col, pt[pt_i][0]) + EPSILON
                coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
                significance = coord_to - coord_from
                # the color at center point is the same as the average of color of segment part
                coord_center = coord_from + 0.5 * significance

                # adding segments that now may contribute
                while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
                    csg[csg_end] = sg[sg_pointer]
                    sg_pointer += 1
                    csg_end += 1
                # removing segments that will no longer contribute
                csg_i = 0
                while csg_i < csg_end:
                    if csg[csg_i][3] < coord_center:
                        csg[csg_i] = csg[csg_end - 1]
                        csg_end -= 1
                    else:
                        csg_i += 1
                best_csg_i: int = 0
                # PERF_COUNTERS[0] += 1
                if csg_end != 1:
                    # PERF_COUNTERS[1] += 1
                    best_csg_closeness: float = -EPSILON
                    for csg_i in range(csg_end):
                        ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
                        # assert 0.0 <= ip_k <= 1.0
                        closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
                        if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
                            best_csg_closeness = closeness
                            best_csg_i = csg_i
                # getting the color
                col_l: int = int(csg[best_csg_i][2] + EPSILON)
                col_r: int = int(csg[best_csg_i][5] + EPSILON)
                if col_l == col_r:
                    color += original_image[row][col_l] * significance
                else:
                    # PERF_COUNTERS[2] += 1
                    ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
                    color += (original_image[row][col_l] * (1.0 - ip_k) +
                              original_image[row][col_r] * ip_k
                              ) * significance
                pt_i += 1
            derived_image[row][col] = np.asarray(color, dtype=np.uint8)
    # print(PERF_COUNTERS)
    print("derived_image shape:",derived_image.shape)
    return derived_image




depthmap = read_image("depthmap.jpg", if_depthmap=True)
origin = read_image("origin.jpg")
result = create_stereoimages(origin, depthmap, divergence=5, separation=0.0, modes=None,
                        stereo_balance=0.0, stereo_offset_exponent=12.0, fill_technique='polylines_sharp')

print("result:",result)