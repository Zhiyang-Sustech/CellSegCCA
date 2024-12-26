import numpy as np
import cv2
import skimage.io
import scipy.io as sio
from scipy import ndimage


# 可视化5种细胞的mask
index = [21, 66, 201, 274, 345, 452, 786, 868, 1938]
masks = np.load("D:\HuHongke\Data\PanNuke2\Masks\masks.npy", allow_pickle=True)
for item in index:
    for j in range(5):
        inst_map = np.zeros(masks[0, :, :, 2].shape, dtype=int)
        inst_map[masks[item,:,:,j] > 0] = 200
        cv2.imwrite("mask_" + str(item) + "_" + str(j) + ".png", cv2.cvtColor(inst_map.astype(np.uint8), cv2.COLOR_RGB2BGR))


# 将图像切割成512×512小patch
# image_path ="D:\HuHongke\mask-rcnn-test\\train20230405T1251_10_chenwendong_new_1\overlay\MOUSE KIDNEY_FFPE_LEICA56_GLASS_20X_2.png"
# original_image = skimage.io.imread(image_path)
# w = 512
# p = int(original_image.shape[0]/w)
# count = 1
# for i in range(p):
#     for j in range(p):
#         cv2.imwrite("D:\HuHongke\Data\TianRuijun\XuYanfen_20x\\20x_1\mini_c2\\20x_1_c1_" + str(count) + ".jpg", cv2.cvtColor(original_image[w*i:w*(i+1), w*j:w*(j+1), :], cv2.COLOR_RGB2BGR))
#         count += 1



# 可视化分割结果中的10个目标，并保存为图像
# mask = sio.loadmat("D:\HuHongke\mask-rcnn-test\\train20230405T1251_10_chenwendong_new_3\mat\MOUSE BRAIN FFPE_20X_2.mat")
# mask = (mask['mask']).astype("int32")
# image_path = "D:\HuHongke\Data\ChenWendong\Figure-4.24\MOUSE BRAIN FFPE_20X_2.PNG"
# original_image = skimage.io.imread(image_path)
# h, w = original_image.shape[:2]
# border = np.zeros(original_image.shape[:2])
# temp = np.zeros(original_image.shape[:2])
# for i in range(1,11):
#     temp[mask==i] = 1
#     outer_border = ndimage.binary_dilation(temp, iterations=4).astype(temp.dtype) - temp
#     border = border + outer_border
# border[border > 0] = 1
# # 贴到原图像
# # 绿色边界
# original_image[:, :, 0][border == 1] = 0
# original_image[:, :, 1][border == 1] = 255
# original_image[:, :, 2][border == 1] = 0
#
# cv2.imwrite("MOUSE BRAIN FFPE_20X_2_overlay.png" , cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
