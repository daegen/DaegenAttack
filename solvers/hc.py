import random
import string
import numpy as np

import time
# from utils.image import l2_norm

def rand_int(low, high, size):
    return np.random.randint(low=low, high=high, size=size)

def choose_pixels(n, p):
    return np.random.choice(n, p, replace=False)

def constrained_mutation(x_t, init_factor=0.15, e=5, single=True):
    if single:
        z_t = x_t.copy()
        # select the index of one single pixel
        pos_t = choose_pixels(len(z_t), 1)[0]
        low = max(0.0, z_t[pos_t] - e)
        high = min(255.0, z_t[pos_t] + e)
        perturbation_t = rand_int(low=low, high=high, size=(1,))[0]
        assert 0. <= perturbation_t <= 255.0
        z_t[pos_t] = perturbation_t
        return z_t
    else:
        z_t = x_t.copy()
        z_size = len(z_t)
        p_size = int(z_size * init_factor)
        p_i = choose_pixels(z_size, p_size)
        p_t = rand_int(low=0, high=256, size=(p_size,))
        np.put(z_t, p_i, p_t)
        return z_t
    

    
def attack_stats(solution_score, time, queries):
    return {
        'solution_score': float(solution_score), 
        'time': time,
        'queries': int(queries),
        # 'l2_norm': float(l2_norm[0]),
        # 'adv_perturbation': float(l2_norm[1])
    }
    
class HillClimbing(object):
    def __init__(self, X_t, max_score=1.0, permutation=False):
        super(HillClimbing, self).__init__()
        assert np.all( 0. <= X_t)
        assert np.all( 255.0 >= X_t)
        self.X_t = X_t
        self.max_score = max_score
        self.evaluation = None
        self.mutation = self.__mutate_solution
        self.pixel_sequence = []
        self.permutation = permutation
        
    def clone_image(self, x_t=None):
        if x_t is not None:
            return np.copy(x_t)
        return np.copy(self.X_t)
    
    def generate_solution(self):
        z_t = self.clone_image()
        # z_size = len(z_t)
        # p_size = int(z_size * self.init_factor)
        # p_i = choose_pixels(z_size, p_size)
        # p_t = rand_int(low=0, high=256, size=(p_size,))
        # np.put(z_t, p_i, p_t)
        return z_t
    
    def __mutate_solution(self, x_t):
        # clone the image
        z_t = self.clone_image(x_t)
        # selects a single pixel to change its value
        # if permutation is true, selects a single pixel randomized pixel queue,
        # this approach 
        # otherwise selects a single pixel at random
        if self.permutation:
            pos_t = self.pixel_sequence[0]
            self.pixel_sequence = self.pixel_sequence[1:]    
        else:
            pos_t = choose_pixels(len(z_t), 1)[0]
        perturbation_t = rand_int(low=1 if z_t[pos_t] == 0 else 0, high=256, size=(1,))[0]
        assert 0. <= perturbation_t <= 255.0
        z_t[pos_t] = perturbation_t
        return z_t

    def mutate_solution(self, x_t, single=True):
        if self.mutation is None:
            return self.__mutate_solution(x_t, single)
        return self.mutation(x_t)
    
    def evaluate(self, solution_t):
        return self.evaluation(self.X_t, solution_t)

    def solve(self, epochs=10000, verbose=True):
        # assert self.mutation is not None, 'Mutation strattegy must be set before optimization starts'
        assert self.evaluation is not None, 'Evaluation strattegy must be set before optimization starts'
        
        if self.permutation:
            self.pixel_sequence = np.random.permutation(self.X_t.shape[0])[:epochs]
        
        solution = self.generate_solution()
        solution_score = self.evaluate(self.clone_image(solution))
        
        e = 0
        while solution_score < self.max_score and e < epochs:
            if verbose and (e % 500 == 0):
                print('Epoch:', e, 'best score so far', solution_score)
            e += 1
            new_solution = self.mutate_solution(solution)
            new_solution_score = self.evaluate(self.clone_image(new_solution))
            
            if (solution_score < new_solution_score):
                solution, solution_score = new_solution, new_solution_score
        return solution_score, solution, e 