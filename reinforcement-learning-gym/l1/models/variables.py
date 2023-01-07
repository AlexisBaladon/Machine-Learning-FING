from enum import Enum

class CELLS(Enum):
    WALL = 1
    EMPTY = 2
    GOAL = 3
    LAVA = 4

#precon: |coord_i| = 2
def norm_one_distance(coord1, coord2):
    dx = abs(coord1[0] - coord2[0])
    dy = abs(coord1[1] - coord2[1])
    res = dx + dy
    return res

class Ninject:
    __instance = None
    def __init__(self, env):
        if Ninject.__instance == None:
            self.agent_info = AgentInfo(env)
            self.map_info = MapInfo()
            self.map_info.agent_info = self.agent_info
            self.agent_info.map_info = self.map_info
            self.instance = self
        
    @staticmethod
    def get_instance(env):
        if(Ninject.__instance == None):
            Ninject.__instance = Ninject(env)
        return Ninject.__instance

    def get_map_info(self):
        return self.map_info

    def get_agent_info(self):
        return self.agent_info

class MapInfo:
    def __init__(self):
        self.gradients = {0: [1,0], 1: [0,1], 2: [-1,0], 3: [0,-1]}
        self.agent_info = None
        self.doors = None

    def __map_tile(self, tile_value):
        if tile_value == -1:
            return CELLS.EMPTY
        elif tile_value == 2:
            return CELLS.WALL
        elif tile_value == 8:
            return CELLS.GOAL
        elif tile_value == 9:
            return CELLS.LAVA
    
    def is_between(self, pos, dir, obs, tile):
        between_walls = self.is_in_front(pos, self.agent_info.rotate_clockwise(dir), obs, tile) == 1 and self.is_in_front(pos, self.agent_info.rotate_counterclockwise(dir), obs, tile) == 1
        return between_walls

    def get_tile(self, coordinate, obs=None):
        value = None
        if len(coordinate) == 2:
            x = coordinate[0]
            y = coordinate[1]
            value = obs[y][x][2]
        elif len(coordinate) == 3:
            value = coordinate[2]
        else:
            raise Exception("Coordinates are incorrectly defined")
        return self.__map_tile(value)

    def get_tile_in_front(self, pos, dir, obs):
        query_pos = [pos[0] + self.gradients[dir][0], pos[1] + self.gradients[dir][1]]
        return self.get_tile(query_pos, obs)

    def get_goal_coord(self, obs):
        for columns in obs:
            for data in columns:
                if self.get_tile(data) == CELLS.GOAL:
                    return [data[1], data[0]]

    def get_distance_to_goal(self, pos, dir, obs):
        green_coord = self.get_goal_coord(obs)
        player_pos = pos
        return norm_one_distance(green_coord, player_pos)

    def is_in_front(self, pos, dir, obs, tile):
        return 1 if self.get_tile_in_front(pos, dir, obs) == tile else 0

    def is_parallel(self, pos, dir, obs, tile):
        clockwise_dir = self.agent_info.rotate_clockwise(dir)
        counterclockwise_dir = self.agent_info.rotate_counterclockwise(dir)
        if(self.is_in_front(pos, clockwise_dir, obs, tile) == 1 or self.is_in_front(pos, counterclockwise_dir, obs, tile) == 1):
            return 1
        else:
            return 0

    def is_in_a_corner(self, pos, dir, obs):
        if self.is_parallel(pos, dir, obs, CELLS.WALL) or self.is_parallel(pos, dir, obs, CELLS.LAVA):
            clockwise_dir = self.agent_info.rotate_clockwise(dir)
            if self.is_parallel(pos, clockwise_dir, obs, CELLS.WALL) or self.is_parallel(pos, clockwise_dir, obs, CELLS.LAVA):
                return 1
        return 0
        
    def is_door_dir(self, pos, dir, obs, tile):
        is_between = self.is_between(pos, dir, obs, tile)
        if is_between:
            clockwise_dir = self.agent_info.rotate_clockwise(dir)
            return 1 - self.is_parallel(pos, clockwise_dir, obs, tile)
        return 0

    def is_door(self, pos, obs, tile):
        dir = 0
        clockwise_dir = self.agent_info.rotate_clockwise(dir)
        return self.is_door_dir(pos, dir, obs, tile) or self.is_door_dir(pos, clockwise_dir, obs, tile)

    def distance_to_door(self, pos, obs):
        best_norm = 50
        if self.doors == None:
            doors = []
            for row in obs:
                for col in row:
                    pos_i = [col[1], col[0]]
                    type = self.__map_tile(col[2])
                    if type == CELLS.EMPTY:
                        lava_door = self.is_door(pos_i, obs, CELLS.LAVA)
                        wall_door = self.is_door(pos_i, obs, CELLS.WALL)
                        if wall_door == 1 or lava_door == 1:
                            doors.append(pos_i)
                            dist = norm_one_distance(pos, pos_i)
                            best_norm = min(dist,best_norm)
            self.doors = doors
        else:
            for door in self.doors:
                dist = norm_one_distance(pos,door)
                best_norm = min(best_norm, dist)
        if self.doors == []:
            best_norm = 0
        return best_norm

class AgentInfo:
    def __init__(self, env):
        self.env = env
        self.map_info = None

    def get_player_dir(self):
        return self.env.agent_dir

    def get_player_pos(self):
        return self.env.agent_pos

    def move_foward(self, player_pos, player_dir, obs):
        if (self.map_info.is_in_front(player_pos,player_dir, obs, CELLS.WALL) == 1):
            return [player_pos[0], player_pos[1]]
        else:
            return [player_pos[0] + self.map_info.gradients[player_dir][0], player_pos[1] + self.map_info.gradients[player_dir][1]]

    def rotate_clockwise(self, player_dir, rotation_value = 1):
        return (player_dir + rotation_value) % 4

    def rotate_counterclockwise(self, player_dir):
        return (player_dir - 1) % 4


class Variables:
    def __init__(self,environment, global_vision = True, vision_range = 5):
        self.env = environment
        inject = Ninject.get_instance(environment)
        self.agent_info = inject.get_agent_info()
        self.map_info = inject.get_map_info()

        self.global_vision = global_vision
        self.vision_range = vision_range

    def __get_distance_to_goal(self, pos, dir, obs, is_moving_fwd = False):
        green_coord = self.map_info.get_goal_coord(obs)
        distance = norm_one_distance(green_coord, pos)
        if not self.global_vision:
            if distance > self.vision_range:
                distance = self.vision_range + 1 - (1 if is_moving_fwd else 0)
        return distance

    def __is_wall_in_front(self, pos, dir, obs):
        return self.map_info.is_in_front(pos,dir,obs, CELLS.WALL)
    
    def __is_lava_in_front(self, pos, dir, obs):
        return self.map_info.is_in_front(pos,dir,obs,CELLS.LAVA)
    
    def __is_goal_in_front(self, pos, dir, obs):
        return self.map_info.is_in_front(pos,dir,obs,CELLS.GOAL)

    def __is_wall_parallel(self, pos, dir, obs):
        return self.map_info.is_parallel(pos,dir,obs, CELLS.WALL) or self.map_info.is_parallel(pos,dir,obs, CELLS.LAVA)
    
    def __is_in_a_corner(self, pos, dir, obs):
        return self.map_info.is_in_a_corner(pos,dir,obs)

    def __is_beside_a_door(self, pos, dir, obs):
        return self.map_info.is_door_dir(pos,dir,obs, CELLS.WALL) or self.map_info.is_door_dir(pos,dir,obs, CELLS.LAVA)

    def __distance_to_door(self, pos, obs, is_moving_fwd = False):
        distance = self.map_info.distance_to_door(pos, obs)
        if not self.global_vision:
            if distance > self.vision_range:
                distance = self.vision_range + 1 - (1 if is_moving_fwd else 0)
        return distance

    def __fell_in_lava(self, pos, obs):
        tile = self.map_info.get_tile(pos,obs)
        if tile == CELLS.LAVA:
            return True
        else:
            return False
        
    def fell_in_lava(self, obs):
        pos = self.agent_info.get_player_pos()
        return self.__fell_in_lava(pos,obs)
        
    def vectorize(self, obs, is_moving_forward, pos = None, dir = None):
        if dir == None:
            pos = self.agent_info.get_player_pos()
            dir = self.agent_info.get_player_dir()
        pos_fwd = pos
        if is_moving_forward:
            pos_fwd = self.agent_info.move_foward(pos,dir,obs)
        distance_to_goal = self.__get_distance_to_goal(pos_fwd, dir, obs, is_moving_forward)
        wall_in_front    = self.__is_wall_in_front(pos_fwd, dir, obs)
        lava_in_front    = self.__is_lava_in_front(pos_fwd, dir, obs)
        goal_in_front    = self.__is_goal_in_front(pos, dir, obs)
        wall_parallel    = self.__is_wall_parallel(pos, dir, obs)
        in_a_corner      = self.__is_in_a_corner(pos, dir, obs)
        door_in_front    = self.__is_beside_a_door(pos_fwd, dir, obs)
        distance_to_door = self.__distance_to_door(pos_fwd, obs, is_moving_forward)
        fell_in_lava     = 1 if self.__fell_in_lava(pos_fwd, obs) else 0 

        return [1, distance_to_goal, wall_in_front, lava_in_front, goal_in_front, wall_parallel, in_a_corner, door_in_front, distance_to_door, fell_in_lava]