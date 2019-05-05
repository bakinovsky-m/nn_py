import csv
import random
import matplotlib.pyplot as plt
from math import exp
import numpy as np

class Dot:
    def __init__(self, cluster, coords):
        self.cluster = cluster

        self.coords = []
        for coord in coords:
            coord = float(coord)
            self.coords.append(coord)

        self.dim = len(coords)

class Node:
    def __init__(self, dim, xy):
        m = []
        for _ in range(dim):
            m.append(random.random())
        self.m = np.array(m)
        self.r = []

        for el in xy:
            self.r.append(el)

        self.dim = dim

    def dist_to_dot(self, dot):
        res = 0
        for i in range(self.dim):
            res += (dot.coords[i] - self.m[i]) ** 2
        return res

    def dist_to_node(self, node):
        res = 0
        for i in range(self.dim):
            res += (node.m[i] - self.m[i]) ** 2
        return res

def alpha(t):
    return 1/((t+1))
    # return 1/((t+1)**(1/4))

def gamma(t):
    return 1/((t+1))
    # return 1/((t+1)**(1/4))

def hci(t, ri, rc):
    dist = ri.dist_to_node(rc)
    # print(dist)
    if dist <= 1.5:
        return alpha(t)
    # if norm <= 1:
    # return 0
    norm = ri.dist_to_node(rc) ** 2
    return alpha(t) * exp(- (norm)/(2*(gamma(t)**2)))

class SOM:
    def __init__(self, dim, node_count):
        self.nodes = []

        cur_x = 0
        cur_y = 0

        self.nodes_map = np.zeros((node_count, node_count), Node)
        for i in range(0, node_count):
            for j in range(0, node_count):
                self.nodes.append(Node(dim, (cur_x, cur_y)))
                self.nodes_map[i][j] = Node(dim, (cur_x, cur_y))
                cur_y += 1
            cur_y = 0
            cur_x += 1
        self.dists = np.zeros((node_count*2, node_count*2))

    def train(self, dots, iter_no):
        ind = random.randint(0, len(dots) - 1)
        dot = dots[ind]
        winner = self.nodes[random.randint(0, len(self.nodes) - 1)]
        winner_dist = 99999

        for n in self.nodes:
            if n.dist_to_dot(dot) <= winner_dist:
                winner = n

        # for n in self.nodes:
        for k in self.nodes_map:
            for n in k:
                h = hci(iter_no, winner, n)
                # for i in range(n.dim):
                    # n.m[i] += h * dot.coords[i] - n.m[i]
                n.m = n.m + h * (dot.coords - n.m)

    def calc_dists(self):
        # old_dists = self.dists
        for i in range(len(self.nodes_map[0])):
            for j in range(len(self.nodes_map[0])):
                n1 = self.nodes_map[i][j]
                neib = {'up':None, 'bottom':None, 'left':None, 'right':None}
                if i != 0:
                    neib['left'] = self.nodes_map[i-1][j]
                    dist = n1.dist_to_node(neib['left'])
                    # print(dist)
                    self.dists[i*2-1][j*2] = dist
                if j != 0:
                    neib['up'] = self.nodes_map[i][j-1]
                    self.dists[i*2][j*2-1] = n1.dist_to_node(neib['up'])
                if i != len(self.nodes_map[0])-1:
                    neib['right'] = self.nodes_map[i+1][j]
                    self.dists[i*2+1][j*2] = n1.dist_to_node(neib['right'])
                if j != len(self.nodes_map[0])-1:
                    neib['bottom'] = self.nodes_map[i][j+1]
                    self.dists[i*2][j*2+1] = n1.dist_to_node(neib['bottom'])
        # diffs = old_dists - self.dists
        # print(diffs)
        return self.dists

dots = []

with open('dataset.csv') as csvf: 
    reader = csv.reader(csvf) 
    for row in reader: 
        coords = row[1:-1]
        dots.append(Dot(row[0], coords))

NODE_COUNT = 10
s = SOM(dots[0].dim, NODE_COUNT)

x_old = []
y_old = []
for n in s.nodes:
    x_old.append(n.r[0]*2+0.5)
    y_old.append(n.r[1]*2+0.5)

for i in range(200):
    print('new loop', i)
    s.train(dots, i)
    dists = s.calc_dists()
    plt.pcolormesh(dists)
    plt.plot(x_old, y_old, 'ro')
    # if i % 100 == 0:
    plt.pause(0.0001)

plt.show()