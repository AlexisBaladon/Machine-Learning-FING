import json
from random import random, choice
from models.l1_model import L1Model
from models.variables import Variables, Ninject
from models.parameters import Parameters
import operator
import os

class G27L1Model(L1Model):
    def __init__(self, environment, **kwargs):
        super().__init__(environment)

        # Values to load/store the model
        self.name = kwargs.get('name', 'g27_model')
        self.file_path = kwargs.get('file_path', './trained_models/') 
        self.vision_range = kwargs.get('vision_range', 3)
        self.global_vision = kwargs.get('global_vision', True)
        self.stochastic = kwargs.get('stochastic', True)
        
        # Variables of the model
        self.vars = Variables(environment, self.global_vision, self.vision_range)
        self.params = Parameters()
        injection = Ninject.get_instance(environment)
        self.agent_info = injection.get_agent_info()
        self.load()
        self.last_random_movements = 0

        # Let's check if cheat mode is on!
        self.cheat_mode = kwargs.get('cheat_mode', False)
        if self.cheat_mode:
            self.cheat_mov = 0
            # Load the sequence of cheat moves, or use F,F,R,F,F by default
            self.cheat_movs = kwargs.get('cheat_movs', [
                self.environment.actions.forward,
                self.environment.actions.forward,
                self.environment.actions.right,
                self.environment.actions.forward,
                self.environment.actions.forward
            ])


    def action(self, observation):
        if self.cheat_mode:
            #Cycle the list of cheat moves
            next_action = self.cheat_movs[self.cheat_mov]
            self.cheat_mov = (self.cheat_mov + 1) % len(self.cheat_movs)
            return next_action
        else:
            #Auxiliaries
            player_pos = self.agent_info.get_player_pos()
            player_dir = self.agent_info.get_player_dir()
            clockwise_player_dir = self.agent_info.rotate_clockwise(player_dir)
            counterclockwise_player_dir = self.agent_info.rotate_counterclockwise(player_dir)

            #Evaluations
            eval_fwd = self.evaluate_pos(observation, player_pos, player_dir, True)
            eval_clockwise_with_move = self.evaluate_pos(observation, player_pos, clockwise_player_dir, True)
            eval_counterclockwise_with_move = self.evaluate_pos(observation, player_pos, counterclockwise_player_dir, True)
            eval_clockwise_without_move = self.evaluate_pos(observation, player_pos, clockwise_player_dir, False)
            eval_counterclockwise_without_move = self.evaluate_pos(observation, player_pos, counterclockwise_player_dir, False)

            #Best evaluation in the same direction (rotate or rotate and move forward)
            eval_clockwise        = max(eval_clockwise_with_move, eval_clockwise_without_move)
            eval_counterclockwise = max(eval_counterclockwise_with_move, eval_counterclockwise_without_move)

            #Best global evaluation
            evals = [eval_fwd, eval_clockwise, eval_counterclockwise] 
            i = evals.index(max(evals)) #Chooses a random index in case of draw

            #Allows similar evaluations to be chosen if stochastic flag is on
            if not self.stochastic:
                idx = i
            else:
                epsilon = 0.005
                j = 0
                res = []
                for v in evals:
                    if abs(v - evals[i]) < (epsilon * abs(evals[i]) if abs(evals[i]) > 0 else epsilon):
                        res.append(j)
                    j += 1

                idx = 0
                if (len(res) > 1):
                    if(self.last_random_movements == 1 or self.last_random_movements == 2):  
                        if(res.count(0) == 1):
                            idx = 0   
                        else:
                            idx = choice(res)
                    else:
                        idx = choice(res)
                    self.last_random_movements = idx
                else:
                    idx = res[0]
                
            if idx == 0:
                return self.environment.actions.forward
            if idx == 1:
                return self.environment.actions.right
            if idx == 2:
                return self.environment.actions.left

    def evaluate_pos(self, obs, pos, dir, isMovingForward):
        """
        Evaluates a certain position according to its direction
        and whether the agent is moving forward or not
        """
    
        values = self.vars.vectorize(obs["image"], isMovingForward, pos, dir)
        weigths = self.params.vectorize()
        if(values[9] == 1):
            return -100
        elif(values[1] == 0):
            return 100
        res = map(operator.mul, values, weigths)
        res = sum(res)
        return res

    def evaluate(self, observation):
        """
        Evaluates the given observation and returns its value.
        """
        pos = self.agent_info.get_player_pos()
        dir = self.agent_info.get_player_dir()
        obs_value = self.evaluate_pos(observation, pos, dir, False) 
        return obs_value


    def load(self):
        """
        Loads the model as json to the configured dir/file_name
        """
        a = os.getcwd()              
        file_name = f"{self.file_path}{self.name}.json" 
        with open(file_name, 'r') as openfile:
            json_config = json.load(openfile)

        vector = json_config['parametros']
        self.params.store(vector)

    def save(self):
        """
        Saves the model as json to the configured dir/file_name
        """     

        model_config = {
            'parametros': self.params.vectorize()
        }
        json_config = json.dumps(model_config, indent=2)

        file_name = f"{self.file_path}{self.name}.json" 
        with open(file_name, "w") as outfile:
            outfile.write(json_config)


    def update(self, **params):
        """
        Updates the model with the given parameters
        """
        self.params.store(params['weights'])

        return self


    def get_params(self):
        return self.params