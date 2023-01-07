from abc import ABC, abstractmethod


class L1Train(ABC):
    """
    Train is done in this class. You should not touch this class, but adapt GXL1Train to your needs.
    """

    def __init__(self, environment,  algorithm, model):
        """
        Initializes a `L1Train` instance. You wont' need to modify this

        Parameters:
        ----------
        environment : gym.Env
            an environment
        algorithm: algos.l1_algo
            the algorithm
        model : models.L1Model
            the model
        """

        # Store parameters
        self.environment = environment
        self.algorithm = algorithm
        self.model = model

    def collect_experience(self, observation):
        """
        Performs one step in the given environment and returs the result of such experience
        
        Parameters:
        ----------
        observation : last observation obtained

        Returns
        -------
        exp :  dictionary with the result of asking an action to the model for the given observation and 
        see the result in the given environment 
        """

        # Choose next action
        start_value = self.model.evaluate(observation)
        action = self.model.action(observation)

        # Perform the action
        next_observation, reward, done, _ = self.environment.step(action)

        # Add advantage and return to experiences
        next_value = self.model.evaluate(next_observation)

        exp = {
            'observation': observation,
            'value': start_value,
            'action': action,
            'next_observation': next_observation,
            'next_value': reward if done else next_value,
            'done': done,
            'reward':reward
        }

        return exp

    @abstractmethod
    def run(self):
        """
        Trains the model. You need to implement it in your the GXL1Train class.
        """
        
        pass
