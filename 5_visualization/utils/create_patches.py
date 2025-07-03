#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **********************************
# Author: kuantian                 #
# Email:  kuantian@tencent.com     #
# **********************************
"""
module docstring.
"""
# standard library
import math
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
import cv2
from utils import seg_utils
import numpy as np
from utils.seg_utils import ContourCheckingFn
from utils.seg_utils import get_contour_check_fn


def is_in_holes(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
            return 1
    return 0


def is_in_contours(cont_check_fn, pt, holes=None, patch_size=256):
    if cont_check_fn(pt):
        if holes is not None:
            return not is_in_holes(holes, pt, patch_size)
        else:
            return 1
    return 0


def process_coord_candidate(coord, contour_holes, ref_patch_size, cont_check_fn):

    if is_in_contours(cont_check_fn, coord, contour_holes, ref_patch_size):
        return coord
    else:
        return None


def process_contour(
        wsi_slide, cont, contour_holes, patch_level, patch_size=256, step_size=256,
        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):

    level_dim = wsi_slide.level_dimensions
    if cont is not None:
        start_x, start_y, w, h = cv2.boundingRect(cont)
    else:
        start_x, start_y, w, h = 0, 0, level_dim[patch_level][0], level_dim[patch_level][1]

    downsample = seg_utils.get_independent_downsample(wsi_slide)[patch_level]
    print(downsample)

    patch_downsample = (int(downsample[0]), int(downsample[1]))
    ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])

    img_w, img_h = level_dim[0]
    if use_padding:
        stop_y = start_y+h
        stop_x = start_x+w
    else:
        stop_y = min(start_y+h, img_h-ref_patch_size[1]+1)
        stop_x = min(start_x+w, img_w-ref_patch_size[0]+1)

    # print("Bounding Box:", start_x, start_y, w, h)
    # print("Contour Area:", cv2.contourArea(cont))

    if bot_right is not None:
        stop_y = min(bot_right[1], stop_y)
        stop_x = min(bot_right[0], stop_x)
    if top_left is not None:
        start_y = max(top_left[1], start_y)
        start_x = max(top_left[0], start_x)

    if bot_right is not None or top_left is not None:
        w, h = stop_x - start_x, stop_y - start_y
        if w <= 0 or h <= 0:
            print("Contour is not in specified ROI, skip")
            return {}, {}
        else:
            print("Adjusted Bounding Box:", start_x, start_y, w, h)

    if isinstance(contour_fn, str):
        cont_check_fn = get_contour_check_fn(
            contour_fn=contour_fn, cont=cont, ref_patch_size=ref_patch_size[0], center_shift=0.5)
    else:
        assert isinstance(contour_fn, ContourCheckingFn)
        cont_check_fn = contour_fn

    step_size_x = step_size * patch_downsample[0]
    step_size_y = step_size * patch_downsample[1]

    x_range = np.arange(start_x, stop_x, step=step_size_x)
    y_range = np.arange(start_y, stop_y, step=step_size_y)

    x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
    coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()


    num_workers = mp.cpu_count()
    if num_workers > 4:
        num_workers = 4
    pool = ThreadPool(num_workers)

    iterable = [(coord, contour_holes, ref_patch_size[0], cont_check_fn) for coord in coord_candidates]
    results = pool.starmap(process_coord_candidate, iterable)
    pool.close()
    results = np.array([result for result in results if result is not None])

    return results


def create_patches_in_tissue(
        wsi_slide, contours_tissue, holes_tissue, patch_level=0, patch_size=256,
        step_size=256, use_padding=True, contour_fn='four_pt_hard'):

    patch_coords = []

    for idx, cont in enumerate(contours_tissue):
        patch_coord = process_contour(
            wsi_slide, cont, holes_tissue[idx], patch_level, patch_size,
            step_size, contour_fn=contour_fn, use_padding=use_padding)

        # discard empty coords
        if patch_coord.shape[0] < 1:
            continue

        patch_coords.append(patch_coord)
    patch_coords = np.concatenate(patch_coords)

    return patch_coords
