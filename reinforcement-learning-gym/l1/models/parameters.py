from turtle import distance


class Parameters:
    def __init__(self):
        self.constant_weight = None
        self.distance_to_goal = None
        self.wall_in_front = None
        self.lava_in_front = None
        self.goal_in_front = None
        self.wall_parallel = None
        self.corner = None
        self.door = None
        self.distance_to_door = None
        self.fell_in_lava = None
        return

    def vectorize(self):
        return [self.constant_weight, self.distance_to_goal, self.wall_in_front, self.lava_in_front, self.goal_in_front, self.wall_parallel, self.corner, self.door, self.distance_to_door, self.fell_in_lava]
    
    def store(self, vector):
        self.constant_weight  = vector[0]
        self.distance_to_goal = vector[1]
        self.wall_in_front    = vector[2]
        self.lava_in_front    = vector[3]
        self.goal_in_front    = vector[4]
        self.wall_parallel    = vector[5]
        self.corner           = vector[6]
        self.door             = vector[7]
        self.distance_to_door = vector[8]
        self.fell_in_lava     = vector[9]