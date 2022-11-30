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

        while found:
            (i, j) = found
            cand = self.__get_cand__(i, j)
            if len(cand) == 1:
                sol[i][j] = self.fixed_sol[i][j] = list(cand)[0]
            else:
                sol[i][j] = np.random.choice(list(cand))
            found = self.find_empty(sol)

        return sol
    
    def count_contradicting_cells(self, i, j, sol=None):
        if sol is None:
            sol = self.fixed_sol
        val = sol[i][j]
        cont = 0
        cont += np.count_nonzero(sol[i, :]==val)-1
        cont += np.count_nonzero(sol[:, j]==val)-1
        cont += np.count_nonzero(sol[int(i/3)*3: int(i/3)*3+3, int(j/3)*3: int(j/3)*3+3]==val)-1
        return cont

    def find_contradicting_cell(self, sol):
        if self.eval_solution(sol)==0:
            return False
        cands_cells = []
        for i in range(9):
            for j in range(9):
                if self.fixed_sol[i][j]==0:
                    cands_cells.append((i, j))
        random.shuffle(cands_cells)
        for cell in cands_cells:
            i, j = cell
            cont = self.count_contradicting_cells(i, j, sol)
            if cont == 0:
                continue
            return cell
        return False


    def get_neighbour_solution(self, sol):
        if self.gen_method == 'mutate':
            for _ in range(self.num_mut):
                found = self.find_contradicting_cell(sol)
                if not found:
                    return sol
                i, j = found
                cand = self.__get_cand__(i, j, sol)
                cand_fixed = self.__get_cand__(i, j)
                if len(cand_fixed) == 0:
                    raise RuntimeError(f'Problem unsolvable, pos ({i} , {j}) is contradicting with every fixed sol.')
                
                val = sol[i][j]
                
                if len(cand_fixed) == 1:
                    sol[i][j] = self.fixed_sol[i][j] = list(cand_fixed)[0]
                else:
                    if len(cand)==0:
                        cand = cand_fixed
                    if len(cand)>1:
                        cand -= {val}
                    sol[i][j] = np.random.choice(list(cand))

        return sol

    def eval_solution(self, sol):
        cost = 0
        for i in range(9):
            cost += (9 - np.unique(sol[:, i]).shape[0]) > 0
            cost += (9 - np.unique(sol[i, :]).shape[0]) > 0
            cost += (9 - np.unique(sol[i%3*3: i%3*3+3, int(i/3)*3: int(i/3)*3+3]).shape[0]) > 0
        return cost

    def find_empty(self, sol):
        for i in range(9):
            for j in range(9):
                if sol[i][j] == 0:
                    return (i, j)

        return False

    def solve_backtrack(self):
        find = self.find_empty(self.fixed_sol)
        if not find:
            return True
        i, j = find
        
        cand = self.__get_cand__(i, j)
        for k in list(cand):
            self.fixed_sol[i][j] = k    
            if self.solve_backtrack():
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
 