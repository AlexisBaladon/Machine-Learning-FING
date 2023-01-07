from abc import ABC, abstractmethod

class L1Model(ABC):
    """
    Class that represent an agent. You will need to modify the GXL1Model to your needs
    """

    def __init__(self, environment):
        # We store the enviroment variable to be able to evaluate and select next actions        
        self.environment = environment

    @abstractmethod
    def action(self, observation):
        """
        Selects and action to perform given the state of the world
        """        
        pass        
        
    @abstractmethod
    def evaluate(self, observation):
        """
        Evaluates the given observation and returns its value.
        """
        pass

    @abstractmethod
    def load(self):
        """
        Loads the model
        """                    
        pass

    @abstractmethod
    def save(self):
        """
        Saves the model
        """                
        pass

    @abstractmethod
    def update(self, **params):
        """
        Updates the model with the given parameters
        """
        return self