__all__ = ['SimulatedAnnealing']

import math
import random
from copy import deepcopy
COOLING_SCHEDULES = ['linear', 'geometric', 'logarithmic', 'exponential', 'linear_inverse']


class SimulatedAnnealing:

    def __init__(self, max_iter=1000, max_iter_per_temp=10,
                 initial_temp=5230.0, final_temp=0.1,
                 cooling_schedule='linear_inverse', cooling_alpha=0.9) -> None:

        self.max_iter = max_iter if max_iter > 0 else 1000
        self.max_iter_per_temp = max_iter_per_temp if max_iter_per_temp > 0 else 10
        self.initial_temp = initial_temp if initial_temp >= 10 else 1000
        self.final_temp = final_temp if 1 >= final_temp > 0 else 0.1
        
        if cooling_schedule not in COOLING_SCHEDULES:
            raise ValueError("Undefined cooling function " + cooling_schedule + ", it must be one of ",
                             str(COOLING_SCHEDULES))
        self.__cooling_schedule = cooling_schedule

        self.__cooling_alpha = cooling_alpha    

        if self.__cooling_schedule == 'linear_inverse' and self.__cooling_alpha <= 0:
            raise ValueError("For cooling function " + self.__cooling_schedule +
                             ", cooling alpha must be greater than 0")
        elif self.__cooling_schedule == 'geometric' and 0.8 > self.__cooling_alpha or self.__cooling_alpha > 0.9:
            raise ValueError("For cooling function " + self.__cooling_schedule +
                             ", cooling alpha must be in range [0.8, 0.9]")

        self.t, self.iter, self.s_best, self.val_best, self.s_cur, self.val_cur, self.problem_obj = [None]*7

    def init_annealing(self, problem_obj=None, stoping_val=None):
        if problem_obj:
            self.problem_obj = problem_obj
        else:
            if not self.problem_obj:
                raise RuntimeError("Problem object need to be set!")

        self.stoping_val = stoping_val
        self.t = self.initial_temp
        self.iter = 1
        self.s_best = self.problem_obj.get_init_solution()
        self.val_best = self.problem_obj.eval_solution(self.s_best)
        self.s_cur = deepcopy(self.s_best)
        self.val_cur = deepcopy(self.val_best)

    def annealing_step(self):
        if not self.problem_obj:
            raise RuntimeError("SimulatedAnnealing problem object is not initialized, call init_annealing()")
        s_cand = self.problem_obj.get_neighbour_solution(self.s_cur)
        val_cand = self.problem_obj.eval_solution(s_cand)
        val_diff = val_cand - self.val_cur
        if val_diff < 0 or random.random() < math.exp(-1*val_diff/self.t):
            self.s_cur = deepcopy(s_cand)
            self.val_cur = deepcopy(val_cand)
            if val_cand < self.val_best:
                self.s_best = deepcopy(s_cand)
                self.val_best = deepcopy(val_cand)
                if not self.stoping_val is None and self.stoping_val == self.val_best:
                    return True

    def update_temperature(self):
        if self.__cooling_schedule == 'linear':
            self.t = self.initial_temp - (self.initial_temp - self.final_temp) * self.iter / self.max_iter
        elif self.__cooling_schedule == 'geometric':
            self.t = self.initial_temp * self.__cooling_alpha ** self.iter
        elif self.__cooling_schedule == 'logarithmic':
            self.t = self.initial_temp / (1 + self.__cooling_alpha * math.log(1 + self.iter))
        elif self.__cooling_schedule == 'exponential':
            self.t = self.initial_temp * math.exp(-1 * self.__cooling_alpha * self.iter ** (1 / self.max_iter))
        elif self.__cooling_schedule == 'linear_inverse':
            self.t = self.initial_temp / (1 + self.__cooling_alpha * self.iter)
        else:
            raise ValueError("Undefined cooling function " + self.__cooling_schedule)

    def run(self, problem_obj=None, stoping_val=None):
        self.init_annealing(problem_obj, stoping_val)
        while self.t > self.final_temp and self.iter <= self.max_iter:
            for _ in range(self.max_iter_per_temp):
                if not self.annealing_step() is None:
                    print('Optimal solution reatched!')
                    return
            self.update_temperature()
            self.iter += 1
