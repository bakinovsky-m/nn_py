import csv
import pygame
import random
import numpy as np
import time
from sklearn import metrics
from sklearn.cluster import KMeans
import hexagons
import matplotlib.pyplot as plt

WINDOW_W=1000
WINDOW_H=1000

class Dot:
    def __init__(self, cluster, coords):
        self.cluster = cluster
        self.c = np.array(coords)

class Node:
    def __init__(self, xy, dim):
        self.c = np.array(xy)

        w = []
        for _ in range(dim):
            w.append(random.uniform(-1, 1))
        self.w = np.array(w)

    def dist(self, dot):
        return np.linalg.norm(self.w - dot.c)

    def dist_to_node(self, node):
        return np.linalg.norm(self.w - node.w)

    def __str__(self):
        return "({},{}):{}".format(self.c[0], self.c[1], self.w)

    def __repr__(self):
        return self.__str__()

class SOM:
    def __init__(self, nrows, ncols, dim):
        nodes = []

        cur_x = 0
        cur_y = 0
        for r in range(nrows):
            new_row = []
            for c in range(ncols):
                new_row.append(Node((cur_x, cur_y), dim))
                cur_x += 1
            nodes.append(new_row)
            cur_x = 0
            cur_y += 1

        self.nodes = np.array(nodes)

        self.train_number = 1

    def train(self, dataset):
        cur_sigma = self.sigma()
        cur_nu = self.nu()
        cur_sigma = (2*cur_sigma**2)
        for _ in range(1):

            dot = np.random.choice(dataset)
            bmu = np.random.choice(np.ravel(self.nodes))
            bmu_dist = bmu.dist(dot)

            ns = np.ravel(self.nodes)
            # start = time.time()
            for n in ns:
                dist = n.dist(dot)
                if dist < bmu_dist:
                    bmu = n
                    bmu_dist = dist
            # end = time.time()
            # print('bmu', end-start)

            # start = time.time()
            for n in np.ravel(self.nodes):
                hij = np.exp(-((n.w-bmu.w)**2)/cur_sigma)
                n.w = n.w + cur_nu * hij * (dot.c - n.w)
            # end = time.time()
            # print('wei', end-start)
            bmu.w = bmu.w + cur_nu * (dot.c - bmu.w)

        self.train_number += 1

    def get_cluters(self, dataset, n_classes):
        nodes = self.nodes.ravel()
        counts = {node:0 for node in nodes}
        for p in dataset:
            d = 9999999
            cur = -1
            for w in nodes:
                dd = w.dist(p)
                if dd < d:
                    d = dd
                    cur = w
            counts[cur] += 1
        import operator
        centers = dict(sorted(counts.items(), key=operator.itemgetter(1), reverse=True)[:3])

        clusters = [-1] * len(dataset)
        counter = 0
        for p in dataset:
            d = 99999
            cur = -1
            for c in range(len(centers)):
                dd = nodes[c].dist(p)
                if dd < d:
                    d = dd
                    cur = c
            clusters[counter] = cur
            counter += 1
        return clusters

    def sigma(self):
        sigma0 = 1
        const = 10
        a = self.train_number/const
        if a > 10:
            a = 10
        return sigma0 * np.exp(-(a))

    def nu(self):
        sigma0 = 1
        const = 2000
        a = self.train_number/const
        if a > 10:
            a = 10
        return sigma0 * np.exp(-(a))

COLS = 15
ROWS = 15
s = SOM(ROWS,COLS,2)

pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
clock = pygame.time.Clock()
FPS = 5

def toc(x, y):
    x1 = int(x * int(WINDOW_W/(ROWS-1)))
    y1 = int(y * int(WINDOW_H/(COLS-1)))
    return (x1, y1)

def toc_ds(x,y):
    x1 = int(x * int(WINDOW_W/2) + int(WINDOW_W/2))
    y1 = int(y * int(WINDOW_H/2) + int(WINDOW_H/2))
    return (x1, y1)

def draw_som(screen, som):
    rows = len(s.nodes)
    cols = len(s.nodes[0])
    for r in range(rows):
        for c in range(cols):
            # pygame.draw.circle(screen, pygame.Color(255,255,255), toc(r,c), 3)
            node = s.nodes[r,c]
            pygame.draw.circle(screen, pygame.Color(255,0,0), toc_ds(node.w[0],node.w[1]), 3)

dataset = []
with open('2d_dataset.csv') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        cluster = row[0]
        coords = [float(c) for c in row[1:-1]]
        dataset.append(Dot(cluster, coords))
    print(len(dataset))

def draw_dataset(screen, dataset):
    for dot in dataset:
        x,y = toc_ds(dot.c[0], dot.c[1])
        pygame.draw.circle(screen, pygame.Color(0,125,125), (x,y), 3)

running = True
counter = 0
fig = plt.figure(1)
while running:
    # clock.tick(FPS)
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        # if e.type == pygame.KEYDOWN:
            # if e.key == pygame.K_RETURN:
                # print('training')
                # for _ in range(100):
                    # s.train(dataset)
    if counter % 100 == 0:
        print("counter", counter)
        screen.fill((0,0,0))
        draw_dataset(screen, dataset)
        draw_som(screen, s)
        # centers = [[0.1,0.2,0.3]]
        centers = [x.c for x in s.nodes.ravel()]
        # weights = [[0.4,.3,.2]]
        weights = [[(x.w[0]+1)/2, (x.w[1]+1)/2, 0.1] for x in s.nodes.ravel()]
        # print(weights)
        hexagons.plot_hex(fig, centers, weights)
        plt.pause(0.001)
        pygame.display.update()
    # if counter == 0:
    #     pygame.image.save(screen, "images/som/som_00000_iter.png")
    # if counter == 100:
    #     pygame.image.save(screen, "images/som/som_00100_iter.png")
    # if counter == 1000:
    #     pygame.image.save(screen, "images/som/som_01000_iter.png")
    # if counter == 5000:
    #     pygame.image.save(screen, "images/som/som_05000_iter.png")
    # if counter == 10000:
    #     pygame.image.save(screen, "images/som/som_10000_iter.png")
    if counter == 15000:
        # pygame.image.save(screen, "images/som/som_15000_iter.png")
        running = False
    # start = time.time()
    s.train(dataset)
    # print(s.get_cluters(dataset, 3))
    # end = time.time()
    # print('training', end - start)
    counter += 1
plt.show()
# print(s.get_cluters(dataset, 3))
dataset_clusters = [x.cluster for x in dataset]
pred_dataset = s.get_cluters(dataset, 3)
# pred_dataset = [x.w for x in s.nodes.ravel()]
# kmeans = KMeans(3).fit(pred_dataset)
# print(kmeans.labels_)
pred_dataset = np.random.randint(0, 2, size=len(dataset))

print('Adjusted Rand score:', metrics.adjusted_rand_score(dataset_clusters, pred_dataset))
print('Mutual Information based score:', metrics.adjusted_mutual_info_score(dataset_clusters, pred_dataset))
print('V-measure score:', metrics.v_measure_score(dataset_clusters, pred_dataset))
print('Fowlkes-Mallows score:', metrics.fowlkes_mallows_score(dataset_clusters, pred_dataset))