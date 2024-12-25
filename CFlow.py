from CTask import *

class CFlowTrainSegModel:
    def __init__(self, config):
        self.config = config

    def run(self):
        taskLoadData = CTaskLoadData()
        dataset_train, dataset_val = taskLoadData.work()

        taskTrainSegModel = CTaskTrainSegModel(self.config)
        taskTrainSegModel.work(dataset_train, dataset_val)

class CFlowTestSegModel:
    def __init__(self, config):
        self.config = config

    def run(self):
        taskTestSegModel = CTaskTestSegModel(self.config)
        taskTestSegModel.work()