import numpy as np
import cv2
from PIL import Image
import os
from scipy.io import savemat

images = np.load("D:\DVPszy\data\dataset\pannuke\\testdata\\3\Images\images.npy", allow_pickle=True)
# types = np.load("D:\HuHongke\Data\PanNuke1\Part1\Images\\types.npy", allow_pickle=True)
masks = np.load("D:\DVPszy\data\dataset\pannuke\\testdata\\3\Masks\masks.npy", allow_pickle=True)

data_output = "D:\DVPszy\data\dataset\pannuke\\testdata\\3\p3myformat\\"
images_path = os.path.join(data_output, "images")
masks_path = os.path.join(data_output, "masks")
if not os.path.exists(data_output):
    os.makedirs(data_output)
if not os.path.exists(images_path):
    os.makedirs(images_path)
if not os.path.exists(masks_path):
    os.makedirs(masks_path)

n_images = images.shape[0]
for i in range(n_images):
    image_path = os.path.join(images_path, str(i) + ".png")
    cv2.imwrite(image_path, cv2.cvtColor(images[i].astype(np.uint8), cv2.COLOR_RGB2BGR))

for i in range(n_images):
    mask = masks[i]
    inst_map = np.zeros(mask.shape[:2], dtype=int)
    type_map = np.zeros(mask.shape[:2], dtype=int)
    count = 1
    for j in range(5):
        type_map[:, :][mask[:, :, j] > 0] = j + 1

        if mask[:, :, j][mask[:, :, j] > 0].shape[0] > 0:
            min = np.min(mask[:, :, j][mask[:, :, j] > 0])
            max = np.max(mask[:, :, j][mask[:, :, j] > 0])
        else:
            continue
        for t in range(int(min), int(max) + 1):
            if mask[:, :, j][mask[:, :, j] == t].shape[0]> 0:
                inst_map[:, :][mask[:, :, j] == t] = count
                count += 1
    mask_path = os.path.join(masks_path, str(i) + ".mat")
    savemat(mask_path, mdict={'inst_map': inst_map, 'type_map': type_map,})


