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
# 3rd part packages
from PIL import Image
import numpy as np
import cv2
from xml.dom import minidom
import h5py
import matplotlib.pyplot as plt
# local source
from utils import wsi_util
from utils import seg_utils

Image.MAX_IMAGE_PIXELS = 933120000


def init_xml(xml_path):
    def create_contour(coord_list):
        return np.array([[[int(float(coord.attributes['X'].value)),
                           int(float(coord.attributes['Y'].value))]] for coord in coord_list], dtype='int32')

    xmldoc = minidom.parse(xml_path)
    annotations = [anno.getElementsByTagName('Coordinate') for anno in xmldoc.getElementsByTagName('Annotation')]
    contours_tumor = [create_contour(coord_list) for coord_list in annotations]
    contours_tumor = sorted(contours_tumor, key=cv2.contourArea, reverse=True)
    return contours_tumor


def draw_grid(img, coord, shape, thickness=2, color=(0, 0, 0, 255)):
    cv2.rectangle(
        img, tuple(np.maximum([0, 0], coord-thickness//2)),
        tuple(coord - thickness//2 + np.array(shape)),
        color, thickness=thickness)
    return img


def draw_map_from_coords(
        canvas, wsi_slide, coords, patch_size,
        vis_level, indices=None, is_draw_grid=True):
    downsamples = wsi_slide.level_downsamples[vis_level]
    if indices is None:
        indices = np.arange(len(coords))
    total = len(indices)

    patch_size = tuple(np.ceil((np.array(patch_size)/np.array(downsamples))).astype(np.int32))
    # print('downscaled patch size: {}x{}'.format(patch_size[0], patch_size[1]))

    for idx in range(total):

        patch_id = indices[idx]
        coord = coords[patch_id]
        patch = np.array(wsi_slide.read_region(tuple(coord), vis_level, patch_size).convert("RGB"))
        coord = np.ceil(coord / downsamples).astype(np.int32)
        canvas_crop_shape = canvas[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0], :3].shape[:2]
        canvas[
            coord[1]:coord[1]+patch_size[1],
            coord[0]:coord[0]+patch_size[0], :3] = patch[:canvas_crop_shape[0], :canvas_crop_shape[1], :]
        if is_draw_grid:
            draw_grid(canvas, coord, patch_size)

    return Image.fromarray(canvas)


def visualize_segmentation(
        wsi_slide, level0_mpp, contours_tissue, holes_tissue,
        color=(0, 255, 0), hole_color=(0, 0, 255), annot_color=(255, 0, 0),
        line_thickness=250, max_size=None, top_left=None, bot_right=None,
        custom_downsample=1, view_slide_only=False,
        number_contours=False, seg_display=True, anno_xml=None, annot_display=True):

    target_mpp = 8.0
    # level0_mpp = wsi_util.get_wsi_mpp(wsi_slide)
    target_downsample = target_mpp / level0_mpp
    vis_level = wsi_slide.get_best_level_for_downsample(target_downsample)
    level_dim = wsi_slide.level_dimensions

    downsample = seg_utils.get_independent_downsample(wsi_slide)[vis_level]
    scale = [1 / downsample[0], 1 / downsample[1]]

    if top_left is not None and bot_right is not None:
        top_left = tuple(top_left)
        bot_right = tuple(bot_right)
        w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
        region_size = (w, h)
    else:
        top_left = (0, 0)
        region_size = level_dim[vis_level]

    img = np.array(wsi_slide.read_region(top_left, vis_level, region_size).convert("RGB"))

    if not view_slide_only:
        offset = tuple(-(np.array(top_left) * scale).astype(int))
        line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
        if contours_tissue is not None and seg_display:
            if not number_contours:
                cv2.drawContours(
                    img, seg_utils.scale_contour_dim(contours_tissue, scale),
                    -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

            else:  # add numbering to each contour
                for idx, cont in enumerate(contours_tissue):
                    contour = np.array(seg_utils.scale_contour_dim(cont, scale))
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / (M["m00"] + 1e-9))
                    cY = int(M["m01"] / (M["m00"] + 1e-9))
                    # draw the contour and put text next to center
                    cv2.drawContours(img, [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                    cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

            for holes in holes_tissue:
                cv2.drawContours(
                    img, seg_utils.scale_contour_dim(holes, scale),
                    -1, hole_color, line_thickness, lineType=cv2.LINE_8)

        if anno_xml is not None and annot_display:
            contours_tumor = init_xml(anno_xml)
            cv2.drawContours(
                img, seg_utils.scale_contour_dim(contours_tumor, scale),
                -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)

    img = Image.fromarray(img)

    w, h = img.size
    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resize_factor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resize_factor), int(h * resize_factor)))

    img = np.array(img)

    return img


def visualize_stitch(
        wsi_slide, level0_mpp, patch_coord, downscale=64, patch_size=256, patch_level=0,
        is_draw_grid=True, bg_color=(0, 0, 0), alpha=-1):

    target_mpp = 8.0
    
    # level0_mpp = wsi_util.get_wsi_mpp(wsi_slide)

    target_downsample = target_mpp / level0_mpp

    vis_level = wsi_slide.get_best_level_for_downsample(target_downsample)
    coords = patch_coord

    print('start stitching ')

    w, h = wsi_slide.level_dimensions[vis_level]

    print('number of patches: {}'.format(len(coords)))

    patch_size = tuple(
        (np.array((patch_size, patch_size)) * wsi_slide.level_downsamples[patch_level]).astype(np.int32))

    if w*h > Image.MAX_IMAGE_PIXELS:
        raise Image.DecompressionBombError("Visualization Downscale %d is too large" % downscale)

    if alpha < 0 or alpha == -1:
        heatmap = Image.new(size=(w, h), mode="RGB", color=bg_color)
    else:
        heatmap = Image.new(size=(w, h), mode="RGBA", color=bg_color + (int(255 * alpha),))

    heatmap = np.array(heatmap)
    heatmap = draw_map_from_coords(
        heatmap, wsi_slide, coords, patch_size, vis_level,
        indices=None, is_draw_grid=is_draw_grid)

    return heatmap


def screen_coords(scores, coords, top_left, bot_right):
    bot_right = np.array(bot_right)
    top_left = np.array(top_left)
    mask = np.logical_and(np.all(coords >= top_left, axis=1), np.all(coords <= bot_right, axis=1))
    scores = scores[mask]
    coords = coords[mask]
    return scores, coords
