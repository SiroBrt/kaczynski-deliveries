import math
import numpy as np
import random
import time

from sklearn.cluster import KMeans


try:
    from ..problem import ProblemInstance, Customer
except ImportError:
    from problem import ProblemInstance, Customer


class Thermite:
    """One of the ants/thermites assigned to the gathering of nodes"""
    
    load: Customer=Customer(
            idx=0,
            x=0,
            y=0,
            demand=0,
            ready_time=0.0,
            due_time=float("inf"),
            service_time=0.0,
            priority=0
        )
    position: (int,int)

    def __init__(self,position):
        self.position = position

    def move(self, limits:[int], step:int=1):
        self.position=(
            min(limits[2],max(limits[0],self.position[0]+random.sample([i for i in range(-step,step)],1)[0])),
            min(limits[3],max(limits[1],self.position[1]+random.sample([i for i in range(-step,step)],1)[0]))
        )

    def load_food(self,food:Customer):
        self.load=food

    def unload_food(self):
        cust = self.load
        self.load=Customer(
            idx=0,
            x=0,
            y=0,
            demand=0,
            ready_time=0.0,
            due_time=float("inf"),
            service_time=0.0,
            priority=0
        )
        return cust


def adjacent(coord):
    result = []
    for h in [-1,0,1]:
        for v in [-1,0,1]:
            result.append((coord[0]+v,coord[1]+h))
    result.remove(coord)
    return result

def find_empty(lista,limits,x,y):
    r = 1
    while True:
        for h in range(-r,r+1):
            if (limits[0] < x+h) or (limits[2] > x+h):
                for v in range(-r,r+1):
                    if (limits[1] < y+v) or (limits[3] > y+v):
                        new_x=x+h
                        new_y=y+v
                        if (new_x,new_y) not in lista:
                            return new_x, new_y
        r += 1



class Board:
    food: dict(coord=(int,int), thing=Customer)
    limits: [int]=[0,0,200,200]
    agents: [Thermite]

    def __init__(self,food,limits,agents):
        self.food=food
        self.limits=limits
        self.agents=agents

    def iterate(self,iterations):
        thermites = [t for t in self.agents]
        for i in range(iterations):
            for t in self.agents:
                coord = t.position
                density = sum([(i in self.food.keys()) for i in adjacent(coord)])/8

                empty = coord not in self.food.keys()
                loaded = (t.load.idx!=0)
                if empty and loaded and (density != 0):
                    cust = t.load
                    affinity = 0
                    for i in adjacent(coord):
                        if i in self.food.keys():
                            affinity += customer_similarity(cust,self.food[i]) 
                    affinity = affinity/(density*8)
                    if random.random() < affinity:
                        self.food[coord] = t.unload_food()
                elif (not empty) and (not loaded) and (random.random()>density):
                    t.load_food(self.food.pop(coord))
                t.move(limits=self.limits)
        
        # Kill all remaining ants
        for t in self.agents:
            coord = t.position
            density = sum([(i in self.food.keys()) for i in adjacent(coord)])/8

            empty = coord not in self.food.keys()
            loaded = (t.load.idx!=0)
            if empty and loaded and (random.random()<density):
                self.food[coord] = t.unload_food()
            elif loaded:
                new_x, new_y = find_empty(self.food.keys(),self.limits,coord[0],coord[1])
                self.food[(new_x,new_y)] = t.unload_food()

        


    def print_board(self):
        for y in range(self.limits[3]-self.limits[1]):
            for x in range(self.limits[2]-self.limits[0]):
                if (x,y) in self.food.keys():
                    print("()",end="")
                else:
                    print("__",end="")
            print("")
        
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def customer_similarity(
    cust_a: Customer,
    cust_b: Customer
    ):
    dist = math.hypot(cust_a.x - cust_b.x, cust_a.y - cust_b.y)
    half = 5 # sets the 0.5 threshold for dist_metric
    dist_metric = 1-sigmoid(dist/half) # inverted logistic for distance


    time_overlap = max(0.0,min(cust_a.due_time,cust_b.due_time)-max(cust_a.ready_time,cust_b.ready_time))
    smol_time = min(cust_a.due_time-cust_a.ready_time,cust_b.due_time-cust_b.ready_time)
    time_metric = time_overlap/smol_time # bigger time metric -> better

    return time_metric/2+dist_metric # average of metrics

def distance_to_cluster(elem,cluster):
    dist = float("inf")
    for i in cluster:
        dist = min(dist, math.hypot(i[0][0]-elem[0][0], i[0][1] - elem[0][1]))
    return dist

def find_clusters(elems, n_clusters):
    # choose centers of clusters
    centers = random.sample(elems,n_clusters)
    result = [] # list [(coor,cust)]
    for i in centers:
        result.append([i])
        elems.remove(i)

    # add elements to clusters alternating
    # other option would be to select the closest to each cluster simultaneoulsy to 
    # avoid "stealing" from a better cluster, but too inefficient
    c = random.sample(range(n_clusters),1)[0]
    while elems:
        min_value = float("inf")
        chosen = elems[0]
        for i in elems:
            d = distance_to_cluster(i,result[c])
            if d < min_value:
                min_value = d
                chosen = i
        result[c].append(chosen)
        elems.remove(chosen)
        c = (c+1)%n_clusters
    cluster = []
    for c in result:
        cluster.append([e[1].idx for e in c])

    return cluster


def thermite_clustering(
    problem: ProblemInstance,
    seed: int = 0,
    n_clusters: int = 1,
    thermites: int=-1,
    iterations: int=-1
    ):
    random.seed(seed)
    if thermites == -1:
        thermites = len(problem.customers)//2

    if iterations == -1:
        iterations = len(problem.customers)*10

    first_limits=[
        int(min([i.x for i in problem.customers])),
        int(min([i.y for i in problem.customers])),
        int(max([i.x for i in problem.customers])),
        int(max([i.y for i in problem.customers]))
        ]

    # we want to have a dense space for thermites to work in
    space = (first_limits[2]-first_limits[0])*(first_limits[3]-first_limits[1])
    ideal_area = len(problem.customers)*2
    shrink_factor= np.floor(np.sqrt(space/ideal_area))

    limits=[
        int(first_limits[0]/shrink_factor),
        int(first_limits[1]/shrink_factor),
        int(first_limits[2]/shrink_factor),
        int(first_limits[3]/shrink_factor)
        ]

    food = {}
    for i in problem.customers:
        x = int(i.x/shrink_factor)
        y = int(i.y/shrink_factor)
        if (x,y) not in food.keys():
            food[(x,y)]=i
        else:
            new_x, new_y = find_empty(food.keys(),limits,x,y)
            food[(new_x,new_y)]=i

    agents = [Thermite(position=(
                        int((limits[2]-limits[0])*random.random())+limits[0],
                        int((limits[3]-limits[1])*random.random())+limits[1]
                    )
                )   for t in range(thermites)]
    
    board = Board(food,limits,agents)

    board.print_board()
    board.iterate(iterations)
    print("working")
    board.print_board()

    # Now we identify the good clusters 
    # should be easy since they are close

    result = find_clusters(list(board.food.items()),n_clusters)

    return result
