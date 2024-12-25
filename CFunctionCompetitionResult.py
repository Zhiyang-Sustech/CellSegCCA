import numpy as np
import pandas as pd
from scipy import ndimage
import scipy.io
from CFunctionUtils import resize


def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def remove_duplicate(mask, threshold=0.7, scores=None):
    if scores is None:
        scores = np.sum(mask, axis=(0, 1))  ## Use size of nucleus as score...
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    flat_mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    for i in np.arange(len(order)):
        mask[:, :, i] = mask[:, :, i] * (flat_mask == order[i])

    new_scores = np.sum(mask, axis=(0, 1))
    diff_pix = scores - new_scores
    reduccion = diff_pix / scores
    # if we have reduced particle size in more than xx percent, remove particle
    ## This only has some effect during the test time augmentation merge
    mask[:, :, reduccion > threshold] = 0
    return mask


def numpy2encoding(predicts, scores=None, threshold=0.7, dilation=False):
    if dilation:
        for i in range(predicts.shape[2]):
            predicts[:, :, i] = ndimage.binary_dilation(predicts[:, :, i])
    predicts = remove_duplicate(predicts, threshold=threshold, scores=scores)
    return predicts

def reformat_mask(predicts):
    mask = np.zeros(predicts.shape[:2])
    predicts = predicts.astype(int)
    for i in range(predicts.shape[2]):
        mask = mask + predicts[:,:,i] * (i + 1)
    return mask.astype(int)

def draw_overlay(predicts, rois, original_image, rescale):
    h, w = original_image.shape[:2]
    border = np.zeros(original_image.shape[:2])
    predicts = predicts.astype(int)
    for i in range(predicts.shape[2]):
        outer_border = ndimage.binary_dilation(predicts[:,:,i], iterations=4).astype(predicts[:,:,i].dtype) - predicts[:,:,i]
        border = border + outer_border
    border[border > 0] = 1

    center = np.zeros([round(h * rescale), round(w * rescale)])
    for i in range(rois.shape[0]):
        x = int((rois[i,0] + rois[i,2])/2)
        y = int((rois[i,1] + rois[i,3])/2)
        center[x, y] = 1
    center = ndimage.binary_dilation(center, iterations=2).astype(center.dtype)
    # center = scipy.ndimage.zoom(center, zoom=[1/rescale, 1/rescale], order=0)
    center = np.around(resize(center, original_image.shape[:2]))

    # 贴到原图像
    # 绿色边界
    original_image[:, :, 0][border == 1] = 0
    original_image[:, :, 1][border == 1] = 255
    original_image[:, :, 2][border == 1] = 0
    # 红色中心点
    original_image[:, :, 0][center == 1] = 255
    original_image[:, :, 1][center == 1] = 0
    original_image[:, :, 2][center == 1] = 0
    return original_image

def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({'ImageId': ImageId, 'EncodedPixels': EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])


def draw_overlay_new(mask,original_image): # 用于重新拼合的mask的轮廓绘制，基本上是原来那个画图函数的阉割版
    mask[mask > 0] = 1
    h, w = mask.shape[:2]
    border = np.zeros(mask.shape[:2])
    predicts = mask.astype(int)
    outer_border = ndimage.binary_dilation(predicts, iterations=4).astype(predicts.dtype) - predicts

    border = border + outer_border
    border[border > 0] = 1

    # 贴到原图像
    # 绿色边界
    original_image[:, :, 0][border == 1] = 0
    original_image[:, :, 1][border == 1] = 255
    original_image[:, :, 2][border == 1] = 0

    return original_image