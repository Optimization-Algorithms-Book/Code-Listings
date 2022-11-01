from .__problem_base import ProblemBase
import numpy as np
import random


class Sudoku(ProblemBase):
    def __init__(self, fixed_vals: dict, gen_method=None, loop=True, **gen_method_kargs) -> None:
        super().__init__()
        self.fixed_vals = fixed_vals
        self.reset()
        if gen_method is None:
            gen_method = 'mutate'
        self.gen_method = gen_method
        if gen_method == 'mutate':
            if 'num_mut' not in gen_method_kargs:
                self.num_mut = 1
            else:
                self.num_mut = gen_method_kargs['num_mut']
        self.elapsed_miss1, self.elapsed_cand, self.elapsed_miss2 = [0] * 3 

    def __get_cand__(self, i, j, sol=None):
        if sol is None:
            sol = self.fixed_sol
        cand = set(range(1, 10))
        cand -= set(sol[i, :])
        cand -= set(sol[:, j])
        cand -= set(sol[int(i/3)*3: int(i/3)*3+3, int(j/3)*3: int(j/3)*3+3].flat)
        return cand

    def reset(self):
        self.fixed_sol = np.array(self.fixed_vals, dtype=int)
        
    def get_init_solution(self):
        sol = np.copy(self.fixed_sol)
        found = self.find_empty(sol)
        if not found:
            return sol
        while found:
            (i, j) = found                
            cand = self.__get_cand__(i, j)
            if len(cand) == 1:
                sol[i][j] = self.fixed_sol[i][j] = list(cand)[0]
            else:
                sol[i][j] = np.random.choice(list(cand))
            found = self.find_empty(sol)
        return sol
    
    def find_contradicting_cell(self, i, j, sol):
        val = sol[i][j]
        cont = 0
        cont += np.count_nonzero(sol[i, :]==val)-1
        cont += np.count_nonzero(sol[:, j]==val)-1
        cont += np.count_nonzero(sol[int(i/3)*3: int(i/3)*3+3, int(j/3)*3: int(j/3)*3+3]==val)-1
        return cont

    def find_contradicting_sol(self, sol):
        while 1:
            i = random.randrange(9)
            j = random.randrange(9)
            if self.fixed_sol[i][j]:
                continue
            cont = self.find_contradicting_cell(i, j, sol)
            if cont == 0:
                continue
            cand = self.__get_cand__(i, j, sol)
            cand_fixed = self.__get_cand__(i, j)
            if len(cand_fixed) == 0:
                raise RuntimeError(f'Problem unsolvable, pos ({i} , {j}) is contradicting with every fixed sol.')
            if len(cand)==0 or self.fixed_sol[i][j] not in cand:
                return (i, j), cand, cand_fixed
        return False


    def get_neighbour_solution(self, sol):
        if self.gen_method == 'mutate':
            for _ in range(self.num_mut):
                found = self.find_contradicting_sol(sol)
                if not found:
                    return sol
                i, j = found[0]
                cand = found[1]
                cand_fixed = found[2]
                k = sol[i][j]
                
                if len(cand_fixed) == 1:
                    sol[i][j] = self.fixed_sol[i][j] = list(cand_fixed)[0]
                else:
                    if len(cand)==0:
                        cand = cand_fixed
                    if len(cand)>1:
                        cand -= {sol[i][j]}
                    sol[i][j] = np.random.choice(list(cand))

        return sol

    def eval_solution(self, sol):
        cost = 0
        for i in range(9):
            cost += 9 - np.unique(sol[:, i]).shape[0]
            cost += 9 - np.unique(sol[i, :]).shape[0]
            cost += 9 - np.unique(sol[i%3*3: i%3*3+3, int(i/3)*3: int(i/3)*3+3]).shape[0]
        return cost

    def find_empty(self, sol):
        for i in range(9):
            for j in range(9):
                if sol[i][j] == 0:
                    return (i, j)

        return False

    def solve_backtrak(self):
        find = self.find_empty(self.fixed_sol)
        if not find:
            return True
        i, j = find
        
        cand = self.__get_cand__(i, j)
        for k in list(cand):
            self.fixed_sol[i][j] = k    
            if self.solve_backtrak():
                return True

        self.fixed_sol[i][j] = 0
        return False
    
    def print(self, sol=None):
        if sol is None:
            sol = self.fixed_sol
        
        for i in range(54):
            chr = '▄' if i == 0 else '▄' if i%18 == 0 else '┬' if i%6==0 else '■'
            print(chr, end='')
        print('▄')
        for i in range(9):
            print('█', end='')
            for j in range(9):
                val = '\033[1m\033[4m\033[91m' + str(sol[i][j]) + '\033[0m' if self.fixed_vals[i][j] else '\033[1m\033[4m\033[92m' + str(sol[i][j]) + '\033[0m' if self.fixed_sol[i][j] != 0 else ' ' if sol[i][j] == 0 else sol[i][j]
                end = '█' if (j+1)%3 == 0 else '|' 
                print('  %s  ' % val, end=end)
            print()
            if i<8:
                for j in range(55):
                    chr = '█' if j==0 else '█' if j==54 else '█' if j%18 == 0 else'┼' if j%6==0 else '■' if (i+1)%3 == 0 else '─'
                    print(chr, end='')
                print()
            else:
                for i in range(54):
                    chr = '▀' if i == 0 else '▀' if i%18 == 0 else '┴' if i%6==0 else '■'
                    print(chr, end='')
                print('▀')
 