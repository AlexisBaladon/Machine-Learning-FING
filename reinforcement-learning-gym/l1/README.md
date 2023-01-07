INSTALL

You will need gym and gym-minigrid

Gym (0.25.1) can be install directly using pip3: pip3 install gym
There are some compatibility issues, so install gym-minigrid manually from: https://github.com/Farama-Foundation/gym-minigrid

    git clone https://github.com/maximecb/gym-minigrid.git
    cd gym-minigrid
    pip3 install -e .

====================================

### Main programs:

- run_me.py, which shows how to read the different variables around mini-grid world (position, objects, rewards, etc.)

- train.py, and example of how to train a model (it will show the results of the dummy implementation).

Example:

To train the agent 100 times, with 25 maximum steps, a vision range of 4, and in an empty room (default map) you should run:

```
py .\train.py --runs 100  --steps 25 --global_vision 0 --vision_range 4
```

(Each flag is appropriately documented in the code)

- test.py, will show the results of the train model, in a GUI if you like

Example:

To train the agent 100 times, with 25 maximum steps, and in a lava room, and in a GUI you should run:

```
py .\test.py --runs 100  --steps 25 --gui 1 --env MiniGrid-LavaCrossingS9N3-v0
```

(Each flag is appropriately documented in the code)

### Load agent values:

Be sure to modify "g27_model.json" weight vector in order to adjust the model before using it. The different positions in the vector mean:

- constant_weight: With no specific interpretation
- distance_to_goal: Distance to green tile
- wall_in_front: Returns true only if a wall is in front of the agent
- lava_in_front: Returns true only if the agent is in front of lava
- goal_in_front: Returns true only if the green tile is in front of the agent
- wall_parallel: Returns true only the agent is moving next to a wall
- in_a_corner: Returns true only if the agent is in front of a corner
- door_in_front: Returns true only if the aget is in front of a door
- distance_to_door: Distance to the closest door
- fell_in_lava: Returns true only if the agent is about to fall in lava (and therefore losing the game).
