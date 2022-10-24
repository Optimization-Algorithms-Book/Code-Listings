from stat import FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
from .__problem_base import ProblemBase
import numpy as np



class Sudoku(ProblemBase):
    def __init__(self, fixed_vals: dict, gen_method, loop=True, **gen_method_kargs) -> None:
        super().__init__()
        self.fixed_vals = fixed_vals
        self.reset()
        self.gen_method = gen_method
        if gen_method == 'mutate':
            if 'num_mut' not in gen_method_kargs:
                self.num_mut = 1
            else:
                self.num_mut = gen_method_kargs['num_mut']

    def __get_cand__(self, i, j):
        cand = set(range(1, 10))
        cand -= set(self.fixed_sol[i, :])
        cand -= set(self.fixed_sol[:, j])
        cand -= set(self.fixed_sol[int(i/3)*3: int(i/3)*3+3, int(j/3)*3: int(j/3)*3+3].flat)
        return cand

    def reset(self):
        self.missing_loc = set()
        for i in range(9):
            for j in range(9):
                self.missing_loc.add((i, j))
        self.missing_loc -= set(self.fixed_vals.keys())
        self.fixed_sol = np.zeros([9, 9], dtype=int)
        for i, j in self.fixed_vals.keys():
            self.fixed_sol[i][j] = self.fixed_vals[(i, j)]

    def get_init_solution(self):
        # sol = np.random.randint(1, 10, [9, 9])
        # for i, j in self.fixed_vals.keys():
        #     sol[i][j] = self.fixed_vals[(i, j)]
        sol = np.copy(self.fixed_sol)
        # for i in range(9):
        #     for j in range(9):
        for (i, j) in list(self.missing_loc):
                
            cand = self.__get_cand__(i, j)
            if len(cand) == 1:
                self.missing_loc -= {(i, j)}
                sol[i][j] = self.fixed_sol[i][j] = list(cand)[0]
            else:
                sol[i][j] = np.random.choice(list(cand))
        return sol
    
    def get_neighbour_solution(self, sol):
        if self.gen_method == 'mutate':
            for rand in np.random.randint(0, len(self.missing_loc), self.num_mut):
                i, j = list(self.missing_loc)[rand]
            # for (i, j) in list(self.missing_loc):

                cand = self.__get_cand__(i, j)
                if len(cand) == 1:
                    self.missing_loc -= {(i, j)}
                    sol[i][j] = self.fixed_sol[i][j] = list(cand)[0]
                else:
                    cand -= {sol[i][j]}
                    sol[i][j] = np.random.choice(list(cand))

        return sol

    def eval_solution(self, sol):
        cost = 0
        for i in range(9):
            cost += 9 - np.unique(sol[:, i]).shape[0]
            cost += 9 - np.unique(sol[i, :]).shape[0]
            cost += 9 - np.unique(sol[i%3*3: i%3*3+3, int(i/3)*3: int(i/3)*3+3]).shape[0]
        cost *= 1000
        return cost
        

    def print(self, sol=None):
        if sol is None:
            sol = self.fixed_sol
    
        for i in range(9):
            if i%3 == 0:
                for j in range(12):
                    print('=====', end='')
            else:
                for j in range(10):
                    print('------', end='')
            print()
            print('||', end='')
            for j in range(9):
                if sol[i][j]:
                    print(' ', sol[i][j], end='  |')
                else:
                    print('   ', end='  |')
                if j%3 == 2:
                    print('||', end='')
            print()
        for j in range(10):
            print('======', end='')


    # def print1(self, sol=None):
    #     if sol is None:
    #         sol = self.fixed_sol
        
    #     for _ in range(55):
    #         print('■', end='')
    #     print()
    #     for i in range(9):
    #         print('█', end='')
    #         for j in range(9):
    #             val = '\033[1m\033[4m\033[91m' + str(sol[i][j]) + '\033[0m' if (i, j) in self.fixed_vals else '\033[1m\033[4m\033[92m' + str(sol[i][j]) + '\033[0m' if self.fixed_sol[i][j] != 0 else ' ' if sol[i][j] == 0 else sol[i][j]
    #             end = '█' if (j+1)%3 == 0 else '│' 
    #             print('  %s  ' % val, end=end)
    #         print()
    #         for j in range(55):
    #             chr = '■' if (i+1)%3 == 0 else '█' if j%18 == 0 else '┼' if j%6==0 else '─' 
    #             print(chr, end='')
    #         print()

    def print1(self, sol=None):
        if sol is None:
            sol = self.fixed_sol
        
        for i in range(54):
            chr = '▄' if i == 0 else '▄' if i%18 == 0 else '┬' if i%6==0 else '■'
            print(chr, end='')
        print('▄')
        for i in range(9):
            print('█', end='')
            for j in range(9):
                val = '\033[1m\033[4m\033[91m' + str(sol[i][j]) + '\033[0m' if (i, j) in self.fixed_vals else '\033[1m\033[4m\033[92m' + str(sol[i][j]) + '\033[0m' if self.fixed_sol[i][j] != 0 else ' ' if sol[i][j] == 0 else sol[i][j]
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
    

    def print2(self, sol=None):
        if sol is None:
            sol = self.fixed_sol
        
        for i in range(54):
            chr = '╔' if i == 0 else '╦' if i%18 == 0 else '┬' if i%6==0 else '═'
            print(chr, end='')
        print('╗')
        for i in range(9):
            print('║', end='')
            for j in range(9):
                val = '\033[1m\033[4m\033[91m' + str(sol[i][j]) + '\033[0m' if (i, j) in self.fixed_vals else '\033[1m\033[4m\033[92m' + str(sol[i][j]) + '\033[0m' if self.fixed_sol[i][j] != 0 else ' ' if sol[i][j] == 0 else sol[i][j]
                end = '║' if (j+1)%3 == 0 else '|' 
                print('  %s  ' % val, end=end)
            print()
            if i<8:
                for j in range(55):
                    chr = '╠' if j==0 else '╣' if j==54 else '╬' if j%18 == 0 else'┼' if j%6==0 else '═' if (i+1)%3 == 0 else '─'
                    print(chr, end='')
                print()
            else:
                for i in range(54):
                    chr = '╚' if i == 0 else '╩' if i%18 == 0 else '┴' if i%6==0 else '═'
                    print(chr, end='')
                print('╝')
            


    def display(self, sol=None):
        if sol is None:
            sol = self.fixed_sol
        for i in range(9):
            if i in [3, 6]:
                print('------+-------+------')
            for j in range(9):
                if j in [3, 6]:
                    print('|', end=" ")
                print(sol[i][j], end=" ")
            print()
