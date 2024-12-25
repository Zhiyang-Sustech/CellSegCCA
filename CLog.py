import logging
import CData_config
import os

class CLogging:
    filePath = ""
    def __init__(self):
        self.filePath = 'log.txt'

    def print_info(self,logInfo, type):

        # 创建一个logger
        logger = logging.getLogger('mylogger')
        logger.setLevel(logging.INFO)

        # 创建一个handler，用于写入日志文件
        if not os.path.exists(CData_config.O_PATH_OUTPUT):
            os.makedirs(CData_config.O_PATH_OUTPUT)
        log_path = os.path.join(CData_config.O_PATH_OUTPUT, self.filePath)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        logger.addHandler(fh)
        logger.addHandler(ch)

        if type == "info":
            logger.info(logInfo)
        elif type == "warning":
            logger.warning(logInfo)
        elif type == "error":
            logger.error(logInfo)

        # 避免重复输出
        logger.removeHandler(fh)
        logger.removeHandler(ch)

