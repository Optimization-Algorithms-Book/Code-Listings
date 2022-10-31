from .__problem_base import ProblemBase
import urllib.request  # the lib that handles the url stuff
import random
import math
import matplotlib.pyplot as plt

class TSP(ProblemBase):
    def __init__(self, load_tsp_file=None, load_tsp_url=None, dists=None, gen_method=None, loop=True, **kargs) -> None:
        super().__init__()
        
        self.cities = []
        if load_tsp_file:
            self.cities = self.load_tsp_from_file(load_tsp_file)    
        elif load_tsp_url:
            self.cities = self.load_tsp_from_url(load_tsp_url)

        if len(self.cities):
            dists = self.eval_distances_from_cities(self.cities)    
        
        self.dists = dists
        self.n = len(dists)
        
        if self.dists == None:
            raise ValueError("Distance matrix with size nxn is required (or tsp file)!")
        
        self.gen_method = 'random_swap'
        if not gen_method is None:
            self.gen_method = gen_method
        
        self.loop = loop
        if not 'init_method' in kargs:
            self.init_method = 'random'
        else:
            self.init_method = kargs['init_method']

        if gen_method == 'random_swap':
            if 'num_swaps' not in kargs:
                self.num_swaps = 1
            else:
                self.num_swaps = kargs['num_swaps']
            if 'swap_wind' not in kargs:
                self.swap_wind = None
            else:
                self.swap_wind = kargs['swap_wind']
        elif gen_method == 'reverse':
            if 'rand_len' in kargs and kargs['rand_len']:
                self.rand_len = True
            else:
                self.rand_len = False
                if 'rev_len' not in kargs:
                    self.rev_len = 2
                else:
                    self.rev_len = kargs['rev_len']
        
    @staticmethod
    def load_tsp_from_file(file_path):
        cities = []
        ignore = True
        with open(file_path, 'r') as reader:
            for line in reader:
                if not ignore:
                    cities.append( [float(_) for _ in line.split()[1:]])
                if line == "NODE_COORD_SECTION":
                    ignore = False
        return cities

    @staticmethod
    def load_tsp_from_url(url):
        cities = []
        ignore = True
        for line in urllib.request.urlopen(url):
            line = line.decode('utf-8').strip()
            if line == 'EOF':
                break
            if not ignore:
                cities.append( [float(_) for _ in line.split()[1:]])
            if line == "NODE_COORD_SECTION":
                ignore = False
        return cities

    @staticmethod
    def eval_distances_from_cities(cities):
        n = len(cities)
        dists = [ [0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                dists[i][j] = dists[j][i] = math.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)
        return dists

    def get_init_solution(self):
        if self.init_method == 'random':
            return random.sample(range(self.n), self.n)
        elif self.init_method == 'greedy':
            mini_dist = 10000000
            min_i, min_j = -1, -1
            for i in range(self.n):
                for j in range(self.n):
                    if i==j:
                        continue
                    if mini_dist > self.dists[i][j]:
                        mini_dist = self.dists[i][j]
                        min_i, min_j = i, j

            sol = [min_i, min_j]
            while len(sol) < self.n:
                mini_dist = 10000000
                min_k = -1
                for k in range(self.n):
                    if k in sol:
                        continue
                    if mini_dist > self.dists[min_j][k]:
                        min_k = k
                        mini_dist = self.dists[min_j][k]
                sol.append(min_k)
                min_j = min_k
            
            return sol


    def get_neighbour_solution(self, sol):
        if self.gen_method == "random_swap":
            for i in range(self.num_swaps):
                c1 = random.randrange(self.n)
                c2 = c1
                while c2 == c1:
                    if not self.swap_wind:
                        c2 = random.randrange(self.n)
                    else:
                        c2 = random.randrange(c1-self.swap_wind, c1+self.swap_wind+1)
                        if c2<0:
                            c2+=self.n
                        elif c2>=self.n:
                            c2-=self.n
                sol[c1], sol[c2] = sol[c2], sol[c1]
            return sol
        elif self.gen_method == 'reverse':
            if self.rand_len:
                l = random.randint(2, self.n - 1)
            else:
                l = self.rev_len
            c1 =  random.randrange(self.n - l)
            sol[c1 : (c1 + l)] = reversed(sol[c1 : (c1 + l)])
            return sol
        elif self.gen_method == 'insert':
            c1 = random.randrange(self.n)
            c2 = c1
            while c2 == c1:
                c2 = random.randrange(self.n)
            sol.insert(c1, sol[c2])
            if c1 < c2:
                c2 += 1
            del sol[c2]
            return sol

    def eval_solution(self, sol):
        cost = 0
        sub = 0 if self.loop else 1
        for i in range(self.n - sub):
            cost += self.dists[sol[i]][sol[(i+1) % self.n]]
        return cost


    def plot(self, path):
        # Unpack the primary TSP path and transform it into a list of ordered
        # coordinates

        if len(self.cities)==0:
            raise RuntimeError("Cannot plot cities as their locations are not provided")
        
        x = []; y = []
        for i in path:
            x.append(self.cities[i][0])
            y.append(self.cities[i][1])

        plt.plot(x, y, 'co')

        # Set a scale for the arrow heads (there should be a reasonable default for this, WTF?)
        a_scale = float(max(x))/float(100)

        # Draw the primary path for the TSP problem
        plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale,
                color ='g', length_includes_head=True)
        for i in range(0,len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                    color = 'g', length_includes_head = True)

        #Set axis too slitghtly larger than the set of x and y
        plt.xlim(min(x)*1.1, max(x)*1.1)
        plt.ylim(min(y)*1.1, max(y)*1.1)
        plt.show()
