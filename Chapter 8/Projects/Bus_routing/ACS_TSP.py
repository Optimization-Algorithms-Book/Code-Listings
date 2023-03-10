import random

class ACS_TSP:
    class Edge:
        def __init__(self, a, b, dist, initial_pheromone):
            self.a = a
            self.b = b
            self.dist = dist
            self.pheromone = initial_pheromone

    class Ant:
        def __init__(self, alpha, beta, num_nodes, edges, r0=0.9):
            self.alpha = alpha
            self.beta = beta
            self.num_nodes = num_nodes
            self.edges = edges
            self.r0 = r0
            self.tour = None
            self.distance = 0.0

        # a function to calculate pheremone levels
        def pheremone(self, level, distance):
            return level ** self.alpha * ((1.0/distance)) ** self.beta

        def select_next_stop(self):

            unvisited_nodes = [stop for stop in range(self.num_nodes) if stop not in self.tour]
            children_pheremones = []
            transition_sum = 0.0

            for stop in unvisited_nodes:
                children_pheremones.append(
                    self.pheremone(
                        self.edges[self.tour[-1]][stop].pheromone,
                        self.edges[self.tour[-1]][stop].dist,
                    )
                )

            random_value = random.uniform(0, 1)

            # when r <= r0, next stop = argmax(children_pheremones)
            if random_value <= self.r0:
                return unvisited_nodes[children_pheremones.index(max(children_pheremones))]

            transition_sum = sum(children_pheremones)

            transition_probability = [
                children_pheremones[i] / transition_sum
                for i in range(len(children_pheremones))
            ]

            # Probabilistically choose a child to explore based weighted by transition probability
            chosen = random.choices(unvisited_nodes, weights=transition_probability, k=1)[0]
            return chosen

        def get_tour(self):
            # start from school
            self.tour = [0]
            for i in range(self.num_nodes - 1):
                self.tour.append(self.select_next_stop())
            return self.tour

        def get_distance(self):
            self.distance = 0.0
            for i in range(self.num_nodes):
                self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].dist
            return self.distance

    def __init__(self, stops, total_num_nodes, colony_size=10, alpha=1.0, beta=3.0, r0 = 0.5,
                 rho=0.1, pheromone_deposit_weight=1.0, initial_pheromone=1.0, iteration=300):
        self.colony_size = colony_size
        self.rho = rho
        self.pheromone_deposit_weight = pheromone_deposit_weight
        self.iteration = iteration
        self.total_num_nodes = total_num_nodes
        self.global_best_tour = None
        self.actual_global_best_tour = None
        self.initial_pheromone = initial_pheromone

        self.global_best_distance = float("inf")
        self.alpha = alpha
        self.beta = beta
        self.r0 = r0
        self.edge_list = None

        self.total_edges = [[None] * self.total_num_nodes for _ in range(self.total_num_nodes)]
        self.edges = []
        self.num_nodes = 0

        for index, row in stops.iterrows():
            self.total_edges[row['i']][row['j']] = self.Edge(row['i'],row['j'], row['dist'] , initial_pheromone)

    def reset_sub_edges(self, edge_list):
        self.edge_list = edge_list
        self.num_nodes = len(edge_list)
        self.edges = [[None] * self.num_nodes for _ in range(self.num_nodes)]
        self.global_best_distance = float("inf")
        self.global_best_tour = None
        self.actual_global_best_tour = None

        for i in range(len(edge_list)):
            for j in range(i, len(edge_list)):
                edge = self.total_edges[edge_list[i]][edge_list[j]]
                edge.pheromone = self.initial_pheromone
                self.edges[i][j] = self.edges[j][i] = edge

    def evaporate_pheromone(self, evaporate):
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                self.edges[i][j].pheromone *= evaporate

    def update_pheromone(self, tour, distance, weight=1.0):
        evaporate = (1 - self.rho)
        self.evaporate_pheromone(evaporate)

        pheromone_to_add = self.pheromone_deposit_weight / distance
        for i in range(self.num_nodes):
            self.edges[tour[i]][tour[(i + 1) % self.num_nodes]].pheromone += weight * pheromone_to_add

    def start(self):
        if(self.num_nodes == 1):
            self.actual_global_best_tour = self.edge_list
            self.global_best_distance = self.edges[0][self.edge_list[0]].dist
            return

        for iter in range(self.iteration):
            local_best_tour = None
            local_best_dist = float("inf")
            for step in range(self.colony_size):
                ant = self.Ant(self.alpha, self.beta, self.num_nodes, self.edges, self.r0)

                # local update pheromone
                cur_tour = ant.get_tour()
                cur_dist = ant.get_distance()
                self.update_pheromone(cur_tour, cur_dist)

                if ant.distance < local_best_dist:
                    local_best_tour = ant.tour
                    local_best_dist = ant.distance
            self.update_pheromone(local_best_tour, local_best_dist)

            if local_best_dist < self.global_best_distance:
                self.global_best_distance = local_best_dist
                self.global_best_tour = local_best_tour

            # global update pheromone
            self.update_pheromone(self.global_best_tour, self.global_best_distance)

        self.convert_to_actual_tour()

    def convert_to_actual_tour(self):
        self.actual_global_best_tour = []
        for i in self.global_best_tour:
            self.actual_global_best_tour.append(self.edge_list[i])

def init(dist, num_nodes, ACS_params):
    return ACS_TSP(stops=dist, total_num_nodes=num_nodes, colony_size=ACS_params['colony_size'],
                   alpha=ACS_params['alpha'], beta=ACS_params['beta'], r0 = ACS_params['r0'],
                   rho=ACS_params['rho'], pheromone_deposit_weight=ACS_params['pheromone_deposit_weight'],
                   initial_pheromone=ACS_params['initial_pheromone'], iteration=ACS_params['iteration'])
