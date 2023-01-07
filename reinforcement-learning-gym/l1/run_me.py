#!/usr/bin/env python3
import argparse
import gym 
# This will register the gym_minigrid envs
from gym_minigrid import envs, wrappers
from gym_minigrid.minigrid import OBJECT_TO_IDX, IDX_TO_OBJECT


AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}
DEFAULT_ENV = 'MiniGrid-FourRooms-v0'


def print_world(image, agent_dir, agent_pos):
    for y_axis in image:
        print("\n\t")
        for cell in y_axis:
            cell_render = AGENT_DIR_TO_STR[agent_dir] if (cell[1]==agent_pos[0] and cell[0]==agent_pos[1]) \
                else IDX_TO_OBJECT[cell[2]][0].upper() if cell[2]>-1 else '_'
            print(cell_render, end='   ')


# Let's get the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default=DEFAULT_ENV, required=False, help=F"Name of the environment (default: {DEFAULT_ENV})")
args = parser.parse_args()

##### Just play around to show what we can do
print (f"\nRunning over {args.env}")
environment = gym.make(args.env, render_mode="human")
environment = wrappers.SymbolicObsWrapper(environment)

obs = environment.reset()
print("\n\nEnvironment loaded")
print("This is our world:\n")
print(obs['image'])

print("\nThis is a list of rows (X is constant). Each row is a list of cells. Each cell has three components.")
print("First and second components are X-axis and Y-axis coordinates (column, row), but inverted. Third component is what there is in the cell (value -1 equals nothing):\n")
for object, idx in OBJECT_TO_IDX.items():
    print (f"\t{object}: {idx}")

#### Check cells
agent_pos = environment.agent_pos
print(f"\nThe agent is now in {agent_pos[0], agent_pos[1]}.")

### So the world looks like
print("\nSo this is our pretty version of our world: ")
print_world(obs['image'], environment.agent_dir, agent_pos)


print("\n\nLet's find what we see in the cells: ")
SOME_CELL_SHIFT = [
    (-1,-1), (0,-1), (1,-1),    
    (-1,0), (0,0),  (1,0),    
    (-1,1), (0,1), (1,1),    
]

for c_shift in SOME_CELL_SHIFT:
    c_pos = (c_shift[0]+agent_pos[0], c_shift[1]+agent_pos[1])
    obs_cell =  obs['image'][c_pos[1], c_pos[0]]
    cell_object = IDX_TO_OBJECT[obs_cell[2]] if obs_cell[2]>-1 else 'nothing'

    print(f"\n\t{c_pos}: in {obs_cell} there is {cell_object}.")
    if c_pos[0]==environment.agent_pos[0] and c_pos[1]==environment.agent_pos[1] and cell_object!='agent':
        print(f"\tThe agent is confused with {cell_object}! :-O")


##### Let's move around
SOME_ACTIONS = [
    environment.actions.right,      # Turn right
    environment.actions.forward,    # One step forward
    environment.actions.forward,    # and another step forward
    environment.actions.left,       # Turn left
    environment.actions.forward,    # Walk two steps
    environment.actions.forward,    # ...
    environment.actions.left,       # Turn left
    environment.actions.forward,    # And go!
]
print("\n\nNow, let's move around...")
print(f"I start in {environment.agent_pos} looking {AGENT_DIR_TO_STR[environment.agent_dir]}")
for step, action in enumerate(SOME_ACTIONS):
    print(f"\n\tStep {step} - Action {action.name}")
    n_obs, reward, done, _ = environment.step(action)
    print(f"\tNow I'm in {environment.agent_pos} looking {AGENT_DIR_TO_STR[environment.agent_dir]}")
    if done:
        break
print(f"\nGame is over with reward {reward}" if done else "\nNo more actions were given, but I could go on...")

print("\nAnd this is the last thing I saw:")
last_cell_obs =  n_obs['image'][environment.agent_pos[1], environment.agent_pos[0]]
last_object = IDX_TO_OBJECT[last_cell_obs[2]] if last_cell_obs[2]>-1 else 'nothing'
print(f"\tObservation in cell {environment.agent_pos}: {last_cell_obs}")
print(f"\tIn that cell there is {last_object}.")

print("\nSo this is our pretty version of our final world: ")
print_world(n_obs['image'], environment.agent_dir, environment.agent_pos)

print("\n\n---- THE END ----\n\n")