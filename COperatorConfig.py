import CData_config
import numpy as np
from CLog import CLogging

def init_config(config):
    config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
    # if CData_config.O_IS_TRAIN:
    #     config.STEPS_PER_EPOCH = 664//config.IMAGES_PER_GPU
    # Input image size
    if config.IMAGE_RESIZE_MODE == "crop":
        config.IMAGE_SHAPE = np.array([config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM, 3])
    else:
        config.IMAGE_SHAPE = np.array([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM, 3])

    # Image meta data length
    # See compose_image_meta() for details
    config.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + config.NUM_CLASSES
    return config

def display_config(is_train, config):
    configurations = "Configurations:"
    # 输出训练或测试的固定参数
    if is_train:
        for a in dir(CData_config):
            if a.startswith("O") or a.startswith("A"):
                configurations += "\n{:30} {}".format(a, getattr(CData_config, a))
    else:
        for a in dir(CData_config):
            if a.startswith("O") or a.startswith("B"):
                configurations += "\n{:30} {}".format(a, getattr(CData_config, a))

    # 输出运行时设置参数
    for a in dir(config):
        if not a.startswith("__") and not callable(getattr(config, a)):
            configurations += "\n{:30} {}".format(a, getattr(config, a))

    configurations += "\n"
    log = CLogging()
    log.print_info("***** Begin *****", "info")
    log.print_info(configurations, "info")