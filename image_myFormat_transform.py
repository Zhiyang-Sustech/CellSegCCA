import numpy as np
import cv2
from PIL import Image
import os
from scipy.io import savemat
from skimage import measure
import skimage.io
import scipy.io as sio
from scipy import ndimage

# mask_image to .mat
# path_output = r"D:\HuHongke\Data\ChenWendong\20230819-labeled\normal\masks"
# path_masks = r"D:\HuHongke\Data\ChenWendong\20230819-labeled\normal\masks_visual"
# masks = os.listdir(path_masks)
#
# for mask_file in masks:
#     base_name = mask_file.split(".png")[0]
#     mask = cv2.imread(os.path.join(path_masks, mask_file), cv2.IMREAD_GRAYSCALE)  # 自动将读到的矩阵灰度值归一化为0-1
#     inst_map = measure.label(mask, connectivity=2)  # 8连通区域标记
#     # for region in measure.regionprops(labels):
#     #     print(type(region))
#
#     mask_path = os.path.join(path_output, base_name + ".mat")
#     savemat(mask_path, mdict={'inst_map': inst_map,})

# .mat to image_mask
path_output = r"D:\HuHongke\Data\PanNuke1_myFormat\masks_visual"
path_masks = r"D:\HuHongke\Data\PanNuke1_myFormat\masks"
path_images = r"D:\HuHongke\Data\PanNuke1_myFormat\images"
masks = os.listdir(path_masks)

for mask_file in masks:
    base_name = mask_file.split(".mat")[0]
    mask = sio.loadmat(os.path.join(path_masks, mask_file))
    mask = (mask['inst_map']).astype("int32")

    image_path = os.path.join(path_images, base_name + '.png')
    original_image = skimage.io.imread(image_path)
    h, w = original_image.shape[:2]
    border = np.zeros(original_image.shape[:2])
    temp = np.zeros(original_image.shape[:2])
    for i in range(1,np.max(mask)+1):
        temp[mask==i] = 1
        outer_border = ndimage.binary_dilation(temp, iterations=4).astype(temp.dtype) - temp
        border = border + outer_border
    border[border > 0] = 1
    # 贴到原图像
    # 绿色边界
    original_image[:, :, 0][border == 1] = 0
    original_image[:, :, 1][border == 1] = 255
    original_image[:, :, 2][border == 1] = 0

    cv2.imwrite(os.path.join(path_output,base_name + '.png'), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

print(count)