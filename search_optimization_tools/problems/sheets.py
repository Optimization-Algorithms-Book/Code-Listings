from .__problem_base import ProblemBase
import urllib.request  # the lib that handles the url stuff
import random
import math
import matplotlib.pyplot as plt

class SHEETS(ProblemBase):
    '''
    init_method             It support two methods of initializing the path, either:
                            'random':       which means the path is generated 
                                            completely random
                            'greedy'        which try to select a sup-optimal initial path
                                            by selecting the pairwise shortest distances 
                                            between citis. This will not leed to the shortest
                                            path but it much better than the random
    '''
    def __init__(self, **kargs) -> None:
        super().__init__()
        
        self.cities = range(7)
        self.letter_mapping = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g'}
        self.dists = {
        (0, 1):3, (0, 2):-1, (0, 3):4, (0, 4):-2, (0, 5):0, (0, 6):1, 
        (1, 0):3, (1, 2):2, (1, 3):3, (1, 4):2, (1, 5):-1, (1, 6):0, 
        (2, 0):-1, (2, 1):2, (2, 3):0, (2, 4):-1, (2, 5):1, (2, 6):2, 
        (3, 0):4, (3, 1):3, (3, 2):0, (3, 4):2, (3, 5):0, (3, 6):-1, 
        (4, 0):-2, (4, 1):2, (4, 2):-1, (4, 3):2, (4, 5):-3, (4, 6):4, 
        (5, 0):0, (5, 1):-1, (5, 2):1, (5, 3):0, (5, 4):-3, (5, 6):3, 
        (6, 0):1, (6, 1):0, (6, 2):2, (6, 3):-1, (6, 4):4, (6, 5):3
        }

        self.n = 7
                    
        if not 'init_method' in kargs:
            self.init_method = 'random'
        else:
            self.init_method = kargs['init_method']



    def get_init_solution(self):
        if self.init_method == 'random':
            return random.sample(range(self.n), self.n)
        
        elif self.init_method == 'greedy':
            max_dist = -float("inf")
            max_i, max_j = -1, -1
            for i in range(self.n):
                for j in range(self.n):
                    if i==j:
                        continue
                    if max_dist < self.dists[(i, j)]:
                        max_dist = self.dists[(i, j)]
                        max_i, max_j = i, j

            sol = [max_i, max_j]
            
            while len(sol) < self.n:
                max_dist = -float("inf")
                max_k = -1
                for k in range(self.n):
                    if k in sol:
                        continue

                    if max_dist < self.dists[(max_j, k)]:
                        max_k = k
                        max_dist = self.dists[(max_j, k)]

                    if max_dist < self.dists[(max_i, k)]:
                        max_k = k
                        max_dist = self.dists[(max_i, k)]

                if self.dists[(max_i, max_k)] > self.dists[(max_j, max_k)]:
                    sol = [max_k] + sol
                else:
                    sol.append(max_k)
                
        return sol


    def get_neighbour_solution(self, sol):
        c1 = random.randrange(self.n)
        c2 = c1
        while c2 == c1:
            c2 = random.randrange(self.n)
        sol[c1], sol[c2] = sol[c2], sol[c1]
        
        swap = (sol[c1], sol[c2]) if sol[c1] < sol[c2] else (sol[c2], sol[c1])
        return sol, swap
        

    def eval_solution(self, sol):
        strength = 0
        for i in range(self.n - 1):
            strength += self.dists[(sol[i], sol[i+1])]
        return strength