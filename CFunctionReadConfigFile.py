import CData_config
import re
import COperatorConfig

class CFunctionConfigFile:
    def _train_seg_file2config(self, filepath, config):
        f_content = open(filepath)
        lines = f_content.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("PATH_TRAIN_DATA"):
                CData_config.A_PATH_TRAIN_DATA = lines[i].split("=")[1].strip()
            elif lines[i].startswith("PATH_VALID_DATA"):
                CData_config.A_PATH_VALID_DATA = lines[i].split("=")[1].strip()
            elif lines[i].startswith("MODEL_PATH"):
                CData_config.O_MODEL_PATH = lines[i].split("=")[1].strip()
            elif lines[i].startswith("NAME"):
                config.NAME = lines[i].split("=")[1].strip()
            elif lines[i].startswith("NUM_CLASSES"):
                config.NUM_CLASSES = int(lines[i].split("=")[1].strip())
            elif lines[i].startswith("IMAGE_RESIZE_MODE"):
                config.IMAGE_RESIZE_MODE = lines[i].split("=")[1].strip()
            elif lines[i].startswith("IMAGE_MIN_DIM"):
                config.IMAGE_MIN_DIM = int(lines[i].split("=")[1])
            elif lines[i].startswith("IMAGE_MAX_DIM"):
                if lines[i].split("=")[1].startswith("F"):
                    config.IMAGE_MAX_DIM = False
                else:
                    config.IMAGE_MAX_DIM = int(lines[i].split("=")[1])
            elif lines[i].startswith("LEARNING_RATE"):
                config.LEARNING_RATE = float(lines[i].split("=")[1])
            elif lines[i].startswith("GPU_COUNT"):
                config.GPU_COUNT = int(lines[i].split("=")[1])
            elif lines[i].startswith("IMAGES_PER_GPU"):
                config.IMAGES_PER_GPU = int(lines[i].split("=")[1])
            elif lines[i].startswith("STEPS_PER_EPOCH"):
                config.STEPS_PER_EPOCH = int(lines[i].split("=")[1])
            elif lines[i].startswith("VALIDATION_STEPS"):
                config.VALIDATION_STEPS = int(lines[i].split("=")[1])
            elif lines[i].startswith("RPN_TRAIN_ANCHORS_PER_IMAGE"):
                config.RPN_TRAIN_ANCHORS_PER_IMAGE = int(lines[i].split("=")[1])
            elif lines[i].startswith("RPN_NMS_THRESHOLD"):
                config.RPN_NMS_THRESHOLD = float(lines[i].split("=")[1])
            elif lines[i].startswith("TRAIN_ROIS_PER_IMAGE"):
                config.TRAIN_ROIS_PER_IMAGE = int(lines[i].split("=")[1])
            elif lines[i].startswith("MAX_GT_INSTANCES"):
                config.MAX_GT_INSTANCES = int(lines[i].split("=")[1])
            elif lines[i].startswith("DETECTION_MAX_INSTANCES"):
                config.DETECTION_MAX_INSTANCES = int(lines[i].split("=")[1])
            elif lines[i].startswith("DETECTION_NMS_THRESHOLD"):
                config.DETECTION_NMS_THRESHOLD = float(lines[i].split("=")[1])
        f_content.close()
        return config

    def _test_seg_file2config(self, filepath, config):
        f_content = open(filepath)
        lines = f_content.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("PATH_TEST_DATA"):
                CData_config.B_PATH_TEST_DATA = lines[i].split("=")[1].strip()
            elif lines[i].startswith("MODEL_PATH"):
                CData_config.O_MODEL_PATH = lines[i].split("=")[1].strip()
            elif lines[i].startswith("TRAIN_SCAN"):
                CData_config.B_TRAIN_SCAN = int(lines[i].split("=")[1])
            elif lines[i].startswith("TEST_SCAN"):
                CData_config.B_TEST_SCAN = int(lines[i].split("=")[1])
            elif lines[i].startswith("NAME"):
                config.NAME = lines[i].split("=")[1].strip()
            elif lines[i].startswith("IMAGE_RESIZE_MODE"):
                config.IMAGE_RESIZE_MODE = lines[i].split("=")[1].strip()
            elif lines[i].startswith("ZOOM"):
                if lines[i].split("=")[1].startswith("F"):
                    config.ZOOM = False
                else:
                    config.ZOOM = float(lines[i].split("=")[1])
            elif lines[i].startswith("ASPECT_RATIO"):
                config.ASPECT_RATIO = float(lines[i].split("=")[1])
            elif lines[i].startswith("MIN_ENLARGE"):
                config.MIN_ENLARGE = float(lines[i].split("=")[1])
            elif lines[i].startswith("IMAGE_MIN_DIM"):
                config.IMAGE_MIN_DIM = int(lines[i].split("=")[1])
            elif lines[i].startswith("IMAGE_MAX_DIM"):
                if lines[i].split("=")[1].startswith("F"):
                    config.IMAGE_MAX_DIM = False
                else:
                    config.IMAGE_MAX_DIM = int(lines[i].split("=")[1])
            elif lines[i].startswith("GPU_COUNT"):
                config.GPU_COUNT = int(lines[i].split("=")[1])
            elif lines[i].startswith("IMAGES_PER_GPU"):
                config.IMAGES_PER_GPU = int(lines[i].split("=")[1])
            elif lines[i].startswith("DETECTION_MIN_CONFIDENCE"):
                config.DETECTION_MIN_CONFIDENCE = float(lines[i].split("=")[1])
            elif lines[i].startswith("RPN_NMS_THRESHOLD"):
                config.RPN_NMS_THRESHOLD = float(lines[i].split("=")[1])
            elif lines[i].startswith("DETECTION_NMS_THRESHOLD"):
                config.DETECTION_NMS_THRESHOLD = float(lines[i].split("=")[1])
            elif lines[i].startswith("DETECTION_MAX_INSTANCES"):
                config.DETECTION_MAX_INSTANCES = int(lines[i].split("=")[1])

            elif lines[i].startswith("AREA_WANTED"):
                config.AREA_WANTED = int(lines[i].split("=")[1])
            elif lines[i].startswith("TRANS_PIXEL"):
                config.TRANS_PIXEL = int(lines[i].split("=")[1])
            elif lines[i].startswith("USE_SPILT"):
                if lines[i].split("=")[1].startswith("F"):
                    config.USE_SPILT = False
                else:
                    config.USE_SPILT = True
            elif lines[i].startswith("BLOCK_RESIZE"):
                config.BLOCK_RESIZE = int(lines[i].split("=")[1])



        f_content.close()
        return config

    def file2config(self, filepath, config):
        # 打开文件读取参数
        f_content = open(filepath)
        lines = f_content.readlines()
        for i in range(len(lines)):
            if lines[i].startswith("PATH_OUTPUT"):
                CData_config.O_PATH_OUTPUT = lines[i].split("=")[1].strip()
                continue
            elif lines[i].startswith("IS_TRAIN"):
                CData_config.O_IS_TRAIN = bool(int(lines[i].split("=")[1]))

        f_content.close()

        if CData_config.O_IS_TRAIN:
            config = self._train_seg_file2config(filepath, config)
            config = COperatorConfig.init_config(config)
            COperatorConfig.display_config(True, config)
        else:
            config = self._test_seg_file2config(filepath, config)
            config = COperatorConfig.init_config(config)
            COperatorConfig.display_config(False, config)

        return config
