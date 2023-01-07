from abc import ABC, abstractmethod


class L1Algo(ABC):
    """
    Class that represent an algorithm. You will need to modify your GXL1Algorithm to your needs
    """

    def __init__(self):
        """
        Initializes a `L1Algo` instance.
        """
        pass



    @abstractmethod
    def fit(self, model, experiences, positions, next_positions):
        """
        Fits the model with the given experiences.
        """
        pass