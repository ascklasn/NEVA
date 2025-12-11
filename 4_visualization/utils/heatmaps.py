import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


def block_blending(wsi_name, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
    print('\ncomputing blend')

    downsample = wsi_name.level_downsamples[vis_level]

    w = img.shape[1]
    h = img.shape[0]
    block_size_x = min(block_size, w)
    block_size_y = min(block_size, h)
    print('using block size: {} x {}'.format(block_size_x, block_size_y))

    shift = top_left  # amount shifted w.r.t. (0,0)

    for x_start in range(top_left[0], bot_right[0], block_size_x):
        for y_start in range(top_left[1], bot_right[1], block_size_y):
            # print(x_start, y_start)

            # 1. convert wsi coordinates to image coordinates via shift and scale
            x_start_img = int((x_start - shift[0]))
            y_start_img = int((y_start - shift[1]))

            # 2. compute end points of blend tile, careful not to go over the edge of the image
            y_end_img = min(h, y_start_img + block_size_y)
            x_end_img = min(w, x_start_img + block_size_x)

            if y_end_img == y_start_img or x_end_img == x_start_img:
                continue
            # print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))

            # 3. fetch blend block and size
            blend_block = img[y_start_img:y_end_img, x_start_img:x_end_img]
            blend_block_size = (x_end_img - x_start_img, y_end_img - y_start_img)

            if not blank_canvas:
                # 4. read actual wsi block as canvas block
                pt = (int(x_start * downsample), int(y_start * downsample))
                canvas = np.array(wsi_name.read_region(pt, vis_level, blend_block_size))

            else:
                # 4. OR create blank canvas block
                canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255, 255, 255,0)))

            # 5. blend color block and canvas block
            img_overlay = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
            img[y_start_img:y_end_img, x_start_img:x_end_img] = img_overlay
    return img


def visHeatmap(wsi_name, scores, coords, vis_level=-1,
               patch_size=(256, 256),
               blank_canvas=False, alpha=0.4,
               blur=False, overlap=0,
               binarize=False, thresh=0.5,
               max_size=None,
               custom_downsample=1,
               cmap='coolwarm'):
    """

    Args:
        scores (numpy array of float): Attention scores
        coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
        vis_level (int): WSI pyramid level to visualize
        patch_size (tuple of int): Patch dimensions (relative to lvl 0)
        blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
        canvas_color (tuple of uint8): Canvas color
        alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
        blur (bool): apply gaussian blurring
        overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
        use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
        convert_to_percentiles (bool): whether to convert attention scores to percentiles
        binarize (bool): only display patches > threshold
        threshold (float): binarization threshold
        max_size (int): Maximum canvas size (clip if goes over)
        custom_downsample (int): additionally downscale the heatmap by specified factor
        cmap (str): name of matplotlib colormap to use
        
    """
    
    level_dim = wsi_name.level_dimensions

    downsample = wsi_name.level_downsamples[vis_level]
    scale = [1 / downsample, 1 / downsample]  # Scaling from 0 to desired level

    if len(scores.shape) == 2:
        scores = scores.flatten()

    if binarize:
        if thresh < 0:
            threshold = 1.0 / len(scores)
        else:
            threshold = thresh
    else:
        threshold = 0.0

    ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
    region_size = level_dim[vis_level]   # 热立图的大小

    top_left = (0, 0)
    bot_right = level_dim[vis_level]

    w, h = region_size

    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
    coords = np.ceil(coords * np.array(scale)).astype(int)

    print('\ncreating heatmap for: ')
    print('top_left: ', top_left, 'bot_right: ', bot_right)
    print('w: {}, h: {}'.format(w, h))
    print('scaled patch size: ', patch_size)

    ######## calculate the heatmap of raw attention scores (before colormap)
    # by accumulating scores over overlapped regions ######

    # heatmap overlay: tracks attention score over each pixel of heatmap
    # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
    # region_size 是热力图的大小    flip是为了将坐标从(x,y)转换为(y,x)，因为numpy的数组是按行优先存储的
    overlay = np.full(np.flip(region_size), 0).astype(float)  # Initialize the overlay matrix with zeros
    counter = np.full(np.flip(region_size), 0).astype(np.uint16)    # Initialize the counter matrix with zeros

    count = 0
    for idx in range(len(coords)):
        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            if binarize:
                score = 1.0
                count += 1
        else:
            score = 0.0

        # accumulate attention for cancer detection
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += score
        # accumulate counter
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1

    if binarize:
        print('\nbinarized tiles based on cutoff of {}'.format(threshold))
        print('identified {}/{} patches as positive'.format(count, len(coords)))

    # fetch attended region and average accumulated attention
    zero_mask = counter == 0

    if binarize:
        overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
    else:
        overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
    del counter

    if blur:
        overlay = cv2.GaussianBlur(overlay, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    if not blank_canvas:
        # downsample original image and use as canvas
        # img = np.array(wsi_name.read_region(top_left, vis_level, region_size).convert("RGB"))
        img = np.array(wsi_name.read_region(top_left, vis_level, region_size))
    else:
        # use blank canvas
        # img = np.array(Image.new(size=region_size, mode="RGBA", color=(255, 255, 255)))
        # Create a blank RGBA canvas with the specified region size and a transparent background
        img = np.array(Image.new(size=region_size, mode="RGBA", color=(255, 255, 255, 0)))

    print('\ncomputing heatmap image')
    print('total of {} patches'.format(len(coords)))
    twenty_percent_chunk = max(1, int(len(coords) * 0.2))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for idx in range(len(coords)):
        if (idx + 1) % twenty_percent_chunk == 0:
            print('progress: {}/{}'.format(idx, len(coords)))

        score = scores[idx]
        coord = coords[idx]
        if score >= threshold:
            # attention block
            raw_block = overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]]

            # color block (cmap applied to attention block)
            # color_block = (cmap(raw_block) * 255)[:, :, :3].astype(np.uint8)
            color_block = (cmap(raw_block) * 255).astype(np.uint8)

            # save raw digits
            # color_block = raw_block * 255
            # color_block = np.repeat(color_block[:, :, None], 3, axis=2)

            # copy over entire color block
            img_block = color_block

            # rewrite image block
            img[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] = img_block.copy()

    # return Image.fromarray(img) #overlay
    print('Done')
    del overlay

    if blur:
        img = cv2.GaussianBlur(img, tuple((patch_size * (1 - overlap)).astype(int) * 2 + 1), 0)

    print(f"before blending: {img.shape} ")

    if alpha < 1.0:
        img = block_blending(wsi_name, img, vis_level, top_left, bot_right, alpha=alpha,
                             blank_canvas=blank_canvas, block_size=patch_size[0])

    img = Image.fromarray(img)

    # img.save("./figs/heatmap.jpg")

    w, h = img.size

    if custom_downsample > 1:
        img = img.resize((int(w / custom_downsample), int(h / custom_downsample)))

    if max_size is not None and (w > max_size or h > max_size):
        resizeFactor = max_size / w if w > h else max_size / h
        img = img.resize((int(w * resizeFactor), int(h * resizeFactor)))

    return img
