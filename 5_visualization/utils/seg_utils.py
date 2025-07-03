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
from multiprocessing import Manager
# 3rd part packages
import numpy as np
import cv2
import openslide
# local source
from utils import wsi_util


def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0:
            continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def get_independent_downsample(wsi_slide):
    level_downsamples = []
    dim_0 = wsi_slide.level_dimensions[0]

    for downsample, dim in zip(wsi_slide.level_downsamples, wsi_slide.level_dimensions):
        estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
        if estimated_downsample != (downsample, downsample):
            level_downsamples.append(estimated_downsample)
        else:
            level_downsamples.append((downsample, downsample))

    return level_downsamples


def scale_contour_dim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]


def scale_holes_dim(contours, scale):
    return [[np.array(hole * scale, dtype='int32') for hole in holes] for holes in contours]


class ContourCheckingFn(object):
    # Defining __call__ method
    def __call__(self, pt):
        raise NotImplementedError


class IsInContourV1(ContourCheckingFn):
    def __init__(self, contour):
        self.cont = contour

    def __call__(self, pt):
        return 1
        #
        # if cv2.pointPolygonTest(self.cont, (int(pt[0]), int(pt[1])), False) >= 0:
        #     return 1
        # else:
        #     return 0


class IsInContourV2(ContourCheckingFn):
    def __init__(self, contour, patch_size):
        self.cont = contour
        self.patch_size = patch_size

    def __call__(self, pt):
        if cv2.pointPolygonTest(
                self.cont,
                (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2),
                False) >= 0:
            return 1
        else:
            return 0


# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class IsInContourV3Easy(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            if cv2.pointPolygonTest(self.cont, (int(points[0]), int(points[1])), False) >= 0:
                return 1
        return 0


# Hard version of 4pt contour checking function - all 4 points need to be in the contour for test to pass
class IsInContourV3Hard(ContourCheckingFn):
    def __init__(self, contour, patch_size, center_shift=0.5):
        self.cont = contour
        self.patch_size = patch_size
        self.shift = int(patch_size // 2 * center_shift)

    def __call__(self, pt):
        center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
        if self.shift > 0:
            all_points = [(center[0] - self.shift, center[1] - self.shift),
                          (center[0] + self.shift, center[1] + self.shift),
                          (center[0] + self.shift, center[1] - self.shift),
                          (center[0] - self.shift, center[1] + self.shift)
                          ]
        else:
            all_points = [center]

        for points in all_points:
            points = (int(points[0]), int(points[1]))
            if cv2.pointPolygonTest(self.cont, points, False) < 0:
                return 0
        return 1


def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = IsInContourV3Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt':
        cont_check_fn = IsInContourV3Easy(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'center':
        cont_check_fn = IsInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = IsInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn


def segment_tissue(wsi_slide, level0_mpp, sthresh=20, sthresh_up=255, mthresh=7, close=0, use_otsu=False,
        filter_params=None, ref_patch_size=512):
    
    target_mpp = 8.0
    # level0_mpp = wsi_util.get_wsi_mpp(wsi_slide)
    target_downsample = target_mpp / level0_mpp
    seg_level = wsi_slide.get_best_level_for_downsample(target_downsample)

    best_w = wsi_slide.level_dimensions[seg_level][0]
    best_h = wsi_slide.level_dimensions[seg_level][1]

    img = np.array(wsi_slide.read_region((0, 0), seg_level, (best_w, best_h)))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

    # Thresholding
    if use_otsu:
        _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    else:
        _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    # Morphological closing
    if close > 0:
        kernel = np.ones((close, close), np.uint8)
        img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    scale = get_independent_downsample(wsi_slide)[seg_level]
    scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale[0] * scale[1]))
    filter_params = filter_params.copy()
    filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
    filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

    # Find and filter contours
    contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
    if filter_params:
        foreground_contours, hole_contours = filter_contours(
            contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

    contours_tissue = scale_contour_dim(foreground_contours, scale)
    holes_tissue = scale_holes_dim(hole_contours, scale)

    # exclude_ids = [0,7,9]
    contour_ids = set(np.arange(len(contours_tissue)))
    contours_tissue = [contours_tissue[i] for i in contour_ids]
    holes_tissue = [holes_tissue[i] for i in contour_ids]

    return contours_tissue, holes_tissue

