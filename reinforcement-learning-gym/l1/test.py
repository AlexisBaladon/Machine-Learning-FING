#!/usr/bin/env python3

import argparse
from xmlrpc.client import Boolean
from models.g27_l1_model import G27L1Model
from models.variables import Variables
import gym 
import time

# This will register the gym_minigrid envs
from gym_minigrid import wrappers


DEFAULT_ENV = 'MiniGrid-Empty-8x8-v0'

# Let's get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=False, help=F"Model name)")
parser.add_argument("--env", type=str, default=DEFAULT_ENV, required=False, help=F"Name of the environment (default: {DEFAULT_ENV})")
parser.add_argument("--runs", type=int, default=2, required=False, help="Number of experiment runs (default: 2)")
parser.add_argument("--steps", type=int, default=20, required=False, help="Number of max steps per run (default: 20)")
parser.add_argument("--seed", type=int, default=-1, required=False, help="Env seed (default: None)")
parser.add_argument("--gui", type=Boolean, default=False, required=False, help="Show GUI (default: False)")
parser.add_argument("--cheat", type=Boolean, default=False, required=False, help="Use a fixed list of actions (default: False)")
parser.add_argument("--global_vision", type=int, default=True, required=False, help="Used to activate agent's global vision")
parser.add_argument("--vision_range", type=int, default=5, required=False, help="If global_vision == 0, it determines the length of the agent's vision")
args = parser.parse_args()

print (f"\n\n********** Testing over {args.env} for {args.runs} runs with {args.steps} max steps **********\n\n")

# Create the environment
seed = None if args.seed == -1 else args.seed
environment = gym.make(args.env, render_mode="human")
environment.max_steps = args.steps
environment = wrappers.SymbolicObsWrapper(environment)


m_kargs = {'cheat_mode': args.cheat,
            'vision_range': args.vision_range,
            'global_vision': args.global_vision}
if args.model:
    m_kargs['model'] = args.model
trained_model = G27L1Model(environment, **m_kargs)
trained_model.load()
variables = Variables(environment, m_kargs["global_vision"], m_kargs["vision_range"])
cant = 0
for i_run in range(0, args.runs):
    obs = environment.reset()[0]
    reward = 0

    for i_step in range(0,args.steps):
        next_action = trained_model.action(obs)
        obs, reward, done, info, _ = environment.step(next_action)

        if args.gui:
            environment.render()
            time.sleep(0.1)

        if done:
            cant = cant + (1 if not variables.fell_in_lava(obs['image']) else 0)
            break
    
    variables.map_info.doors = None
    if args.gui:
        time.sleep(2)

print("Victory rate: ",cant,"/",args.runs)
