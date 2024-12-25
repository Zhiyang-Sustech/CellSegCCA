from CFunctionReadConfigFile import CFunctionConfigFile
from CFlow import CFlowTrainSegModel, CFlowTestSegModel
import CData_config
from CData_config import Config
from CLog import CLogging

class CStaff:
    def __init__(self, inputArgv):
        self.argv = inputArgv
        self.config = Config()

    def start(self):
        self._captainRunFlow()

    def _captainRunFlow(self):
        configFile = CFunctionConfigFile()
        self.config = configFile.file2config(self.argv[1], self.config)

        if CData_config.O_IS_TRAIN:
            CData_config.O_FLOW = "TrainModel"

            flowTrainSegModel = CFlowTrainSegModel(self.config)
            flowTrainSegModel.run()

        elif not CData_config.O_IS_TRAIN:
            CData_config.O_FLOW = "TestModel"

            flowTestSegModel = CFlowTestSegModel(self.config)
            flowTestSegModel.run()

        log = CLogging()
        log.print_info("***** End *****", "info")