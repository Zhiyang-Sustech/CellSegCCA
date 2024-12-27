import cv2
seed = 123
from scipy import ndimage

import numpy as np
np.random.seed(seed)

import tensorflow as tf

#import tensorflow._api.v2.compat.v1 as tf
#tf.disable_v2_behavior()


tf.random.set_seed(seed)

#tf.random.set_random_seed(seed)

#tf.random.set_seed(args.seed)

import random
random.seed(seed)
import os
import sys
import time
import traceback
from imgaug import augmenters as iaa
import skimage.io
from skimage import img_as_ubyte
from CLog import CLogging
import CData_config
from CTool import *
import os
import traceback
from CFunctionPanNuke import PanNukeDataset
# from CFunctionCoNSeP import CoNSePDataset
import CFunctionModel as modellib
import pandas as pd
from PIL import Image
import scipy.io

import CFunctionCompetitionResult as f
from CFunctionUtils import resize

import SplitAndMergePic

class CTaskLoadData:
    def work(self):
        log = CLogging()
        log.print_info("###Load Data", "info")
        try:
            dataset_train = PanNukeDataset()
            dataset_train.load_dataset(CData_config.A_PATH_TRAIN_DATA)
            dataset_train.prepare()

            # # Validation dataset, same as training.. will use pad64 on this one
            dataset_val = PanNukeDataset()
            dataset_val.load_dataset(CData_config.A_PATH_VALID_DATA)
            dataset_val.prepare()
        except:
            log.print_info(traceback.format_exc(), "error")
            exit(1)

        log.print_info("###Load Data Finished", "info")

        return dataset_train, dataset_val

class CTaskTrainSegModel:
    def __init__(self, config):
        self.config = config

    def work(self, dataset_train, dataset_val):
        log = CLogging()
        log.print_info("###Train Model", "info")

        MODEL_DIR = os.path.join(CData_config.O_PATH_OUTPUT, "logs")

        # Local path to trained weights file
        ## https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
        # COCO_MODEL_PATH = os.path.join(CData_config.O_ROOT_DIR, CData_config.O_MODEL_PATH)
        try:
            model = modellib.MaskRCNN(mode="training", config=self.config,
                              model_dir=MODEL_DIR)

            # model.load_weights(COCO_MODEL_PATH, by_name=True,
            #                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
            #                             "mrcnn_bbox", "mrcnn_mask"])
            model.keras_model.load_weights(CData_config.O_MODEL_PATH, by_name=True,)

            import time
            start_time = time.time()

            ## Augment True will perform flipud fliplr and 90 degree rotations on the 512x512 images

            # Image augmentation
            # http://imgaug.readthedocs.io/en/latest/source/augmenters.html

            ## This should be the equivalente version of my augmentations using the imgaug library
            ## However, there are subtle differences so I keep my implementation
            augmentation = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.OneOf([iaa.Affine(rotate=0),
                           iaa.Affine(rotate=90),
                           iaa.Affine(rotate=180),
                           iaa.Affine(rotate=270)]),
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-10, 10))),
                iaa.Add((-15, 15), per_channel=1)
            ])

            # augmentation=False
            augment = False
            model.train(dataset_train, dataset_val,
                        learning_rate=self.config.LEARNING_RATE/3,
                        epochs=75,
                        augmentation=augmentation,
                        layers="all")

            end_time = time.time()
            ellapsed_time = (end_time - start_time) / 3600
            log.print_info("Time: " + str(ellapsed_time), "info")

            model_path = os.path.join(model.log_dir, 'test_mode1.h5')
            model.keras_model.save_weights(model_path)
            log.print_info("Model saved at: " + model_path, "info")
        except:
            log.print_info(traceback.format_exc(), "error")
            exit(1)
        log.print_info("###Train Model Finished", "info")


class CTaskTestSegModel:
    def __init__(self, config):
        self.config = config

    def work(self):
        log = CLogging()
        log.print_info("###Test Model", "info")

        MODEL_DIR = os.path.join(CData_config.O_PATH_OUTPUT, "logs")
        try:
            import time
            start_time = time.time()

            # Recreate the model in inference mode
            model = modellib.MaskRCNN(mode="inference",
                                      config=self.config,
                                      model_dir=MODEL_DIR)
            # model.load_weights(CData_config.O_MODEL_PATH, by_name=True)
            model.keras_model.load_weights(CData_config.O_MODEL_PATH, by_name=True, )

            mat_dict = {}

            rm_n_mkdir(CData_config.O_PATH_OUTPUT + "\\mat\\")
            rm_n_mkdir(CData_config.O_PATH_OUTPUT + "\\overlay\\")

            images = os.listdir(CData_config.B_PATH_TEST_DATA)
            rescale = CData_config.B_TRAIN_SCAN / CData_config.B_TEST_SCAN
            for i in np.arange(len(images)):
                log.print_info('Start detect '+ str(i) +'  ' + images[i], "info")
                ##Set seeds for each image, just in case..
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)

                ## Load the image
                image_path = os.path.join(CData_config.B_PATH_TEST_DATA, images[i])
                big_image = skimage.io.imread(image_path)

                # 初始化面积统计的所有参数
                area = 0
                masks_list = []
                overlay_list = []
                area_save = 0
                area_list = []
                #area_wanted = self.config.AREA_WANTED
                #trans_pixel = self.config.TRANS_PIXEL
                #Use_Spilt = self.config.USE_SPILT
                #area_wanted_pixel = area_wanted * trans_pixel
                #block_resize = self.config.BLOCK_RESIZE
                small_image_count = 0

                #测试用功能，直接关闭
                Use_Spilt = False
                area_wanted = 0
                



                if Use_Spilt == True:
                    small_image = SplitAndMergePic.split_pic(big_image, block_resize)


                else:
                    small_image = []
                    small_image.append(big_image)




                while small_image_count < len(small_image):
                    original_image = small_image[small_image_count]

                    small_image_count = small_image_count + 1


                    ####################################################################
                    ## This is needed for the stage 2 image that has only one channel
                    if len(original_image.shape) < 3:
                        original_image = img_as_ubyte(original_image)
                        original_image = np.expand_dims(original_image, 2)
                        original_image = original_image[:, :, [0, 0, 0]]  # flip r and b

                    original_image = original_image[:, :, :3]
                    masks = np.zeros(original_image.shape[:2]).astype(int)

                    # 若测试集扫描倍率与训练集扫描倍率不等，则缩放测试集图像再测试
                    if rescale != 1:
                        h, w = original_image.shape[:2]
                        resize_image = resize(original_image, (round(h * rescale), round(w * rescale)), preserve_range=True)
                        results = model.detect([resize_image], verbose=0)
                    else:
                        results = model.detect([original_image], verbose=0)

                    ## Proccess prediction into rle
                    pred_masks = results[0]['masks']
                    scores_masks = results[0]['scores']
                    class_ids = results[0]['class_ids']
                    rois = results[0]['rois']

                    if len(class_ids):  ## Some objects are detected
                        predicts = f.numpy2encoding(pred_masks, scores=scores_masks, dilation=True)
                        # predicts = scipy.ndimage.zoom(predicts, zoom=[1/rescale, 1/rescale, 1], order=0)
                        predicts = np.around(resize(predicts, [original_image.shape[0], original_image.shape[1], predicts.shape[2]]))

                        masks = f.reformat_mask(predicts)

                        # 后面这段是对分割之后的面积进行统计

                        if Use_Spilt == False and area_wanted == 0:
                            overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)

                        if (Use_Spilt == True and area_wanted == 0) or (area_wanted != 0): #不启动分割


                            list_scores = scores_masks.tolist()
                            num = len(list_scores)
                            list_masks = masks.tolist()
                            test = 1

                            pixel_count = 0
                            round_count = 0
                            while test <= num and (area < area_wanted_pixel or (area >= area_wanted_pixel and round_count == 0)):
                                for x in list_masks:
                                    for y in x:
                                        if y == test:
                                            pixel_count = pixel_count + 1

                                area = area + pixel_count
                                area_list.append(pixel_count)


                                if area >= area_wanted_pixel and round_count == 0:
                                    area_save = area
                                    round_count = round_count + 1
                                    break
                                elif area >= area_wanted_pixel and round_count != 0:
                                    round_count = round_count + 1
                                    break
                                else:
                                    pixel_count = 0
                                    test = test + 1
                            #print(area)

                            if Use_Spilt == True and area_wanted == 0:
                                #masks = f.reformat_mask(predicts)
                                #overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)

                                masks_list.append(masks)

                                #overlay_list.append(overlay_image)

                            if area_wanted != 0:

                                if Use_Spilt == False and area < area_wanted_pixel: # 不分割，面积不足
                                    print('Low Area, Only %.1f' %(area/trans_pixel))
                                    overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)

                                elif Use_Spilt == False and area >= area_wanted_pixel: # 不分割，面积溢出

                                    #del_num = test
                                    del_range = []
                                    #while del_num <= (len(list_scores)-1):
                                        #del_range.append(del_num)
                                        #del_num = del_num + 1

                                    pred_masks = np.delete(pred_masks, del_range, 2)
                                    scores_masks = np.delete(scores_masks, del_range, 0)
                                    class_ids = np.delete(class_ids, del_range, 0)
                                    rois = np.delete(rois, del_range, 0)

                                    predicts = f.numpy2encoding(pred_masks, scores=scores_masks, dilation=True)
                                    # predicts = scipy.ndimage.zoom(predicts, zoom=[1/rescale, 1/rescale, 1], order=0)
                                    predicts = np.around(resize(predicts, [original_image.shape[0], original_image.shape[1], predicts.shape[2]]))

                                    masks = f.reformat_mask(predicts)

                                    overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)

                                    print('Area got %.1f' %(area / trans_pixel))

                                #elif (Use_Spilt == True and area < area_wanted_pixel) or (Use_Spilt == True and area >= area_wanted_pixel and round_count == 1 and test == num): #启动分割，每块小区域都统计面积，直至达到目标，达标之后的区域仍检查但结果归零便于最后生成图像

                                    #masks_list.append(masks)
                                    #overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)
                                    #overlay_list.append(overlay_image)

                                    #masks = SplitAndMergePic.merge_pic(masks_list,block_resize)
                                    #overlay_image =SplitAndMergePic.split_pic(overlay_list)

                                elif (Use_Spilt == True and area < area_wanted_pixel) or (Use_Spilt == True and area >= area_wanted_pixel and round_count == 1):
                                    del_num = test
                                    del_range = []
                                    while del_num <= (len(list_scores) - 1):
                                        del_range.append(del_num)
                                        del_num = del_num + 1

                                    pred_masks = np.delete(pred_masks, del_range, 2)
                                    scores_masks = np.delete(scores_masks, del_range, 0)
                                    class_ids = np.delete(class_ids, del_range, 0)
                                    rois = np.delete(rois, del_range, 0)

                                    predicts = f.numpy2encoding(pred_masks, scores=scores_masks, dilation=True)
                                    # predicts = scipy.ndimage.zoom(predicts, zoom=[1/rescale, 1/rescale, 1], order=0)
                                    predicts = np.around(resize(predicts, [original_image.shape[0], original_image.shape[1],
                                                                           predicts.shape[2]]))

                                    masks = f.reformat_mask(predicts)
                                    overlay_image = f.draw_overlay(predicts, rois, original_image, rescale)

                                    masks_list.append(masks)
                                    overlay_list.append(overlay_image)

                                elif (Use_Spilt == True and area >= area_wanted_pixel and round_count > 1) or (Use_Spilt == True and area >= area_wanted_pixel and small_image_count > 1):
                                    masks = masks * 0 #检测到的masks归零
                                    overlay_image = original_image
                                    masks_list.append(masks)
                                    overlay_list.append(overlay_image)






                        # if rescale != 1:    # 恢复原尺寸
                        #     overlay_image = overlay_image.astype(np.uint8)
                        #     cv2.imwrite("%s/overlay_rescale/%s.png" % (CData_config.O_PATH_OUTPUT, images[i].split('.')[0]), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
                        #     masks = scipy.ndimage.zoom(masks, zoom=[1/rescale, 1/rescale], order=0)
                        #     count = 0
                        #     for j in range(1, 200):
                        #         if np.sum(masks[masks == j]) > 0:
                        #             count += 1
                        #     print("transform number:", count)
                        #     overlay_image = resize(overlay_image, (h, w), preserve_range=True).astype(np.uint8)
                    elif not len(class_ids) and Use_Spilt == False:
                        log.print_info('No particles detected' + images[i], "info")
                        overlay_image = original_image

                if Use_Spilt == True:
                    new_masks_list = []
                    for mask in masks_list:
                        mask = ndimage.binary_dilation(mask, iterations=5)
                        new_masks_list.append(mask)

                    masks = SplitAndMergePic.merge_pic(new_masks_list,block_resize)
                    overlay_image = f.draw_overlay_new(masks,big_image)
                    #overlay_image = SplitAndMergePic.merge_pic(overlay_list,block_resize)
                    area_got = area_save/trans_pixel
                    if area_save >= area_wanted_pixel and area_wanted_pixel != 0:
                        print('Area got %.1f' %(area_got))
                    if area < area_wanted_pixel and area_wanted_pixel != 0:
                        print('Low Area, Only %.1f' %(area / trans_pixel))


                cv2.imwrite("%s/overlay/%s.png" % (CData_config.O_PATH_OUTPUT, images[i].split('.')[0]), cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

                mat_dict['mask'] = masks
                scipy.io.savemat("%s/mat/%s.mat" % (CData_config.O_PATH_OUTPUT, images[i].split('.')[0]), mat_dict)





            end_time = time.time()
            ellapsed_time = (end_time - start_time) / 3600
            log.print_info('Time required to test ' + str(ellapsed_time) +  'hours', "info")
        except:
            log.print_info(traceback.format_exc(), "error")
            exit(1)
        log.print_info("###Test Model Finished", "info")
