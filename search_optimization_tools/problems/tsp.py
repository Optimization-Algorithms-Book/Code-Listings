from .__problem_base import ProblemBase
import random


class TSP(ProblemBase):
    def __init__(self, dists, gen_method, **gen_method_kargs) -> None:
        super().__init__()
        self.n = len(dists)
        self.dists = dists
        self.gen_method = gen_method
        if gen_method == 'random_swap':
            if 'num_swaps' not in gen_method_kargs:
                self.num_swaps = 1
            else:
                self.num_swaps = gen_method_kargs['num_swaps']

    def get_init_solution(self):
        return random.sample(range(self.n), self.n)
    
    def get_neighbour_solution(self, sol):
        if self.gen_method == "random_swap":
            for i in range(self.num_swaps):
                c1 = random.randrange(self.n)
                c2 = c1
                while c2 == c1:
                    c2 = random.randrange(self.n)
                sol[c1], sol[c2] = sol[c2], sol[c1]
            return sol

    def eval_solution(self, sol):
        cost = 0
        for i in range(self.n):
            cost += self.dists[sol[i]][sol[(i+1) % self.n]]
        return cost
