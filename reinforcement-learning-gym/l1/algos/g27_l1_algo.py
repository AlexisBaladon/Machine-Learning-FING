from algos.TestFactory import TestFactory
from algos.l1_algo import L1Algo

class G27L1Algo(L1Algo):
    def __init__(self, **kwargs):
        super().__init__()
        self.penalty = kwargs.get("step_penalty", 0.9)
        self.base_learning_rate = kwargs.get("learning_rate", 0.1)
        self.learning_rate_decay = kwargs.get("learning_rate_decay", 1)
        self.max_iterations = kwargs.get("max_runs", 1)
        self.iterations = 0

        #Testing
        self.test_number = kwargs.get("test", -1)
        self.test_dir = kwargs.get("test_dir", "")
        self.test = TestFactory().createTest(self.test_number, self.max_iterations)

    # We adjust the learning rate with exponential decay. Other models may be used
    def __learning_rate(self):
        return self.base_learning_rate * (self.learning_rate_decay ** self.iterations)

    def fit(self, model, experiences, positions, next_positions):
        experiences.reverse()
        weights = model.params.vectorize()
        i = 0
        for experience in experiences:
            variables = model.vars.vectorize(experience["observation"]["image"], False, positions[i][0],positions[i][1])
            j = 0
            v_train = experience['reward'] if i == 0 else model.evaluate_pos(experience["observation"],next_positions[i][0],next_positions[i][1], False) 
            v_techo = model.evaluate_pos(experience["observation"],positions[i][0],positions[i][1], False)
            change_rate = self.penalty*v_train - v_techo
            for w in weights:
                w_j = w + self.__learning_rate()*(change_rate)*variables[j]
                weights[j] = w_j
                j = j + 1
            i = i + 1

            model.update(weights=weights)
                
        #Testing
        if self.test_number != -1:
            self.test.update_data(iteration=self.iterations, weights=weights)
            if self.iterations == self.max_iterations - 1:
                self.test.persist(files_dir=self.test_dir)
        
        self.iterations = self.iterations + 1
        return model