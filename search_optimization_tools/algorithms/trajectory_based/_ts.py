__all__ = ['TabuSearch']

import math
import random
from copy import deepcopy
import numpy as np
import collections.abc

class Hashing:
    def __init__(self, v) -> None:
        self.v = v

    def __repr__(self) -> str:
        return self.v.__repr__()

    def __str__(self) -> str:
        return self.v.__str__()
    
    def __getitem__(self, item):
        return self.v[item]
    
    def __hash__(self) -> int:
        if isinstance(self.v, collections.abc.Hashable):
            return hash(self.v)
        if type(self.v).__module__ == np.__name__:
            return hash(tuple(self.v.flat))
        if type(self.v) is set:
            return hash(tuple(self.v))
        if type(self.v) is dict:
            return hash(tuple(sorted(self.v.keys())))
        if type(self.v) is list:
            return hash(tuple(np.array(self.v).flat))
        raise RuntimeError('Unresolved hashing case!')

class TabuSearch:

    def __init__(self, max_iter=1000, tabu_tenure=1000, neighbor_size=10,
                use_aspiration=True, aspiration_limit=None, use_longterm=False, 
                 debug=0) -> None:

        self.debug = debug
        self.max_iter = max_iter if max_iter and max_iter > 0 else 1000
        self.tabu_tenure = tabu_tenure if tabu_tenure and tabu_tenure > 0 else 1000
        self.neighbor_size = neighbor_size
        self.use_aspiration = use_aspiration
        self.aspiration_limit = aspiration_limit if aspiration_limit and aspiration_limit > 0 else tabu_tenure + 1
        self.use_longterm = use_longterm

    def init_ts(self, problem_obj=None, stoping_val=None, init=None):
        if problem_obj:
            self.problem_obj = problem_obj
        else:
            if not self.problem_obj:
                raise RuntimeError("Problem object need to be set!")

        self.stoping_val = stoping_val
        self.iter = 1
        self.tabu_list = {}
        if not init is None:
            self.s_cur = init
        else:
            self.s_cur = Hashing(self.problem_obj.get_init_solution())
        self.val_cur = self.problem_obj.eval_solution(self.s_cur.v)
        self.s_best, self.val_best = deepcopy(self.s_cur), deepcopy(self.val_cur)
        self.iter_best = 0
        self.s_allbest, self.val_allbest = [None] * 2
        self.iter_all_best = 0
        if self.use_longterm:
            self.total_sol = 0
            self.longterm = {}
        if self.debug>0:
            print(f"Tabu search is initialized:\ncurrent value = {self.val_cur}")
        
    def ts_step(self):
        if not self.problem_obj:
            raise RuntimeError("Tabu search problem object is not initialized, call init_ts()")

        s_cand, val_cand = [None] *2
        i = 0
        while s_cand is None:
            s_cand, val_cand = self.get_best_neighbour(self.s_cur)
            i += 1
            if i>1000:
                if self.debug>0:
                    print(f"Optimal solution so far!: \ncurr iter: {self.iter}, curr best value: {self.val_best}, curr best: sol: {self.s_best}, found at iter: {self.iter_best}")
                raise RuntimeError(f"Search space is too narrow (probably {len(self.tabu_list)}) which are all in the tabu list, try to increase the search space or set stopping value")
        
        # if i > 1:
        #     print(i, len(self.tabu_list))

        
        if val_cand < self.val_best:
            self.s_best = deepcopy(s_cand)
            self.val_best = deepcopy(val_cand)
            self.iter_best = self.iter    
        
        self.s_cur = deepcopy(s_cand)
        self.val_cur = deepcopy(val_cand)
        
        self.tabu_list = {k: v-1 for k,v in self.tabu_list.items() if v>1}

        if self.use_longterm:
            self.total_sol += 1
            if not self.s_cur in self.longterm:
                self.longterm[self.s_cur] = 0      
            self.longterm[self.s_cur] += 1

        self.tabu_list[self.s_cur] = self.tabu_tenure
        
    def get_best_neighbour(self, s_cur):
        s_cands = set()
        while len(s_cands) < self.neighbor_size:
            s_cands.add(deepcopy(Hashing(self.problem_obj.get_neighbour_solution(s_cur.v))))

        s_cands = [k for k in s_cands if not k in self.tabu_list or 
                                                    (self.use_aspiration and 
                                                     self.problem_obj.eval_solution(k.v)<self.val_cur and 
                                                     self.aspiration_limit>self.tabu_list[k])]
        
        if len(s_cands) == 0:
            return None, None

        best_cand = None
        val_best_cand = 1000000000

        for s_cand in s_cands:
            val_cand = self.problem_obj.eval_solution(s_cand.v)
            if (not self.use_longterm or not s_cand in self.longterm or 
                random.random() < self.longterm[s_cand] / self.total_sol) and val_cand < val_best_cand:
                val_best_cand = val_cand
                best_cand = s_cand

        return best_cand, val_best_cand

    
    def run(self, problem_obj=None, stoping_val=None, init=None, repetition=1):
        self.init_ts(problem_obj, stoping_val, init)
        for __ in range(repetition):
            for self.iter in range(1, self.max_iter+1):
                self.ts_step()        
                if self.debug>1:
                    print(f"curr iter: {self.iter}, curr value: {self.val_cur}, curr best value: {self.val_best}, curr best: sol: {self.s_best}, found at iter: {self.iter_best}")

                if not self.stoping_val is None and self.stoping_val == self.val_best:
                    if self.debug>0:
                        print(f"Optimal solution reatched!: \ncurr iter: {self.iter}, curr best value: {self.val_best}, curr best: sol: {self.s_best}, found at iter: {self.iter_best}")
                    return
        
            if self.val_allbest is None or self.val_best < self.val_allbest:
                self.s_allbest = deepcopy(self.s_best)
                self.val_allbest = deepcopy(self.val_best)
                self.iter_all_best = self.iter_best
            if __  < repetition - 1:
                if self.debug>0:
                    print(f'Best solution at rep. {__+1} is:{self.val_best}')
                self.val_best = None
                self.init_ts(problem_obj, stoping_val, self.s_best)
        
        self.s_best = deepcopy(self.s_allbest)
        self.val_best = deepcopy(self.val_allbest)
        self.iter_best = self.iter_all_best
        if self.debug>0:
            print(f"Tabu search is done: \ncurr iter: {self.iter}, curr best value: {self.val_best}, curr best: sol: {self.s_best}, found at iter: {self.iter_best}")
