class NotTrainedError(Exception):
    pass


class FailedTrainingError(Exception):
    pass


class FailedOptimizationError(Exception):
    pass


class TimelimitError(Exception):
    pass


class InvalidConfigError(Exception):
    pass


class InvalidVariableNameError(Exception):
    pass
