#!/usr/bin/env python3

import argparse
from xmlrpc.client import Boolean
from models.g27_l1_model import G27L1Model
from algos.g27_l1_algo import G27L1Algo
from training.g27_l1_train import G27L1Train

import gym 

# This will register the gym_minigrid envs
from gym_minigrid import envs, wrappers

DEFAULT_ENV = 'MiniGrid-Empty-8x8-v0'

# Let's get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default=DEFAULT_ENV, required=False, help=F"Name of the environment (default: {DEFAULT_ENV})")
parser.add_argument("--runs", type=int, default=1, required=False, help="Number of experiment runs (default: 2)")
parser.add_argument("--steps", type=int, default=20, required=False, help="Number of max steps per run (default: 20)")
parser.add_argument("--seed", type=int, default=-1, required=False, help="Env seed (default: None)")
parser.add_argument("--global_vision", type=Boolean, default=True, required=False, help="Used to activate agent's global vision")
parser.add_argument("--vision_range", type=int, default=5, required=False, help="If global_vision == 0, it determines the length of the agent's vision")
parser.add_argument("--step_penalty", type=float, default=0.9, required=False, help="Penalize each step taken by a factor of the input")
parser.add_argument("--learning_rate", type=float, default=0.1, required=False, help="Determines the learning rate of the algorithm")
parser.add_argument("--learning_rate_decay", type=float, default=1, required=False, help="Exponential decay for the learning rate of successive training iteration")
parser.add_argument("--test", type=int, default=-1, required=False, help="If test == 1, it tests the convergence rate of current run (designed to add more tests in a future).")
parser.add_argument("--test_dir", type=str, default="", required=False, help="If the flag test has the value of a valid test, the obtained information will be saved in this direction")
parser.add_argument("--stochastic", type=int, default=1, required=False, help="If this flag is 0, the evaluation method will be deterministic. Otherwise, it allows similar evaluations to be chosen")

args = parser.parse_args()

print (f"\n\n********** Training over {args.env} for {args.runs} runs with {args.steps} max steps **********\n\n")

seed = None if args.seed == -1 else args.seed

environment = gym.make(args.env, seed=seed, render_mode="human")
environment.max_steps = args.steps
environment = wrappers.SymbolicObsWrapper(environment)

training = G27L1Train(environment, G27L1Algo(step_penalty=args.step_penalty, learning_rate=args.learning_rate, learning_rate_decay=args.learning_rate_decay, test=args.test, test_dir=args.test_dir, max_runs=args.runs), G27L1Model(environment, name="g27_model", stochastic=args.stochastic), max_runs=args.runs, max_steps=args.steps, global_vision=args.global_vision, vision_range=args.vision_range)

trained_model = training.run()
trained_model.save()
