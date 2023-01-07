from abc import ABC, abstractmethod
from models.variables import Variables

from training.l1_train import L1Train

class G27L1Train(L1Train):
    """
    Train is done in this class. You will need to adapt this class to your needs.
    """

    def __init__(self, environment, algorithm, model, **kwargs):
        super().__init__(environment, algorithm, model)

        # We add max_runs and max_steps paramaters
        self.max_runs = kwargs.get('max_runs', 1)
        self.max_steps = kwargs.get('max_steps', 20)
        self.vars = Variables(environment, kwargs["global_vision"], kwargs["vision_range"])

    def run(self):
        """
        Runs max_runs episodes, where the agent performs at most max_steps actions. 
        In this implementation, the model is adjusted after each episode, but feel free to change it.
        """

        # Go for max_runs episodes
        for i_run in range(0, self.max_runs):

            # Restart the environment
            observation = self.environment.reset()

            # Go for max_steps per experiment (actually, this cna be set in the env directly)
            experiences = []
            positions = []
            next_positions = []
            for i_step in range(0, self.max_steps):
                agent_pos = self.environment.agent_pos
                agent_dir = self.environment.agent_dir
                positions.append([agent_pos, agent_dir])
                # Do a step in this world
                experience = self.collect_experience(observation)

                agent_pos = self.environment.agent_pos
                agent_dir = self.environment.agent_dir
                next_positions.append([agent_pos, agent_dir])

                if experience['done']:
                    if self.vars.fell_in_lava(experience["next_observation"]["image"]):
                        experience["next_value"] = -100
                        experience["reward"]     = -100
                        
                    else:
                        experience["next_value"] = 100
                        experience["reward"]     = 100
                    experiences.append(experience)
                    break
                elif i_step == (self.max_steps - 1):
                    experience["next_value"] = 0
                    experience["reward"]     = 0
                # Store the experience for learning
                experiences.append(experience)
                # change the observation state
                observation = experience['next_observation']

            # Let's update the agente after each episode
            positions.reverse()
            next_positions.reverse()
            self.model = self.algorithm.fit(self.model, experiences, positions, next_positions)
            self.vars.map_info.doors = None
        return self.model

        
        


                








