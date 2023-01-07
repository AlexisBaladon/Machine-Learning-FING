from abc import ABC, abstractmethod

class TestFactory():
    def __init__(self, **kwargs):
        super().__init__()
        
    def createTest(self, test_number, max_iterations):
        if test_number == 1:
            return TestNormDistance(max_iterations = max_iterations)
        else:
            return TestNull()

class Test(ABC):
    def __init__(self, **kwargs):
        super().__init__()
        
    @abstractmethod
    def update_data(**kwargs):
        pass

    @abstractmethod
    def persist(**kwargs):
        pass

class TestNull(Test):
    def __init__(self, **kwargs):
        super().__init__()    

    def update_data(self, **kwargs):
        pass

    def persist(self, **kwargs):
        pass

class TestNormDistance(Test):
    def __init__(self, **kwargs):
        super().__init__()
        self.last_weight = None
        self.track_weights = {}
        self.max_iterations = kwargs.get('max_iterations', -1)

    def __norm_two_distance(self, c1, c2):
        import math
        res = 0
        i = 0
        for c in c1:
            res = res + ((c - c2[i])**2)
            i = i + 1
        return math.sqrt(res)

    def update_data(self, **kwargs):
        iteration = kwargs.get('iteration', -1)
        weights = kwargs.get('weights', [])

        if self.last_weight != None:
            self.track_weights[iteration] = self.__norm_two_distance(weights, self.last_weight)
        self.last_weight = weights

    def persist(self, **kwargs):
        import json
        import experiments as exp

        file_dir = kwargs.get('files_dir', '')
        weight_list = list(self.track_weights.values())
        exp.continuous_function("Convergencia de vector de pesos",range(0,len(weight_list)), weight_list, "Iteración", "||Xi - Xi+1||", file_dir)
        with open(f"{file_dir}\graph", "w") as outfile:
            outfile.write(str(json.dumps(self.track_weights)))