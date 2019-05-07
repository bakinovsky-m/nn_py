import csv
import numpy as np
import random
import pygame
WINDOW_W=1000
WINDOW_H=1000

def toc_ds(x,y):
    x1 = int(x * int(WINDOW_W/2) + int(WINDOW_W/2))
    y1 = int(y * int(WINDOW_H/2) + int(WINDOW_H/2))
    return (x1, y1)

def draw_dataset(screen, dataset):
    for dot in dataset:
        x,y = toc_ds(dot.c[0], dot.c[1])
        pygame.draw.circle(screen, pygame.Color(0,125,125), (x,y), 3)

class Dot:
    def __init__(self, cluster, coords):
        self.cluster = cluster
        self.c = np.array(coords)

dataset = []
with open('2d_dataset.csv') as csvf:
    reader = csv.reader(csvf)
    for row in reader:
        cluster = row[0]
        coords = [float(c) for c in row[1:-1]]
        dataset.append(Dot(cluster, coords))
dataset = np.array(dataset)

clusters = {}

R = 0.275

pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
FPS=15
clock = pygame.time.Clock()
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
    clock.tick(FPS)

    screen.fill((0,0,0))
    draw_dataset(screen, dataset)
    pygame.display.update()

    if len(dataset) == 0:
        for cent, nei in clusters.items():
            x,y = toc_ds(cent.c[0],cent.c[1])
            pygame.draw.circle(screen, (0,255,0), (x,y), int(R*WINDOW_H/2))
            for _,n in nei.items():
                x,y = toc_ds(n.c[0],n.c[1])
                pygame.draw.circle(screen, (0,0,255), (x,y), 3)
        pygame.display.update()
        continue


    cur_d_ind = random.randint(0, len(dataset)-1)
    cur_d = dataset[cur_d_ind]
    neibs = {cur_d_ind:cur_d}

    prev_center = Dot(999, (999,999))
    center = None
    center_ind = 0
    neibs_to_delete = {}
    while center != prev_center and running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                print('quitting')
                running = False
                break
        screen.fill((0,0,0))
        draw_dataset(screen, dataset)
        ind = 0
        for d in dataset:
            # dist = np.linalg.norm(cur_d.c - d.c)
            if center == None:
                dist = np.linalg.norm(cur_d.c - d.c)
            else:
                dist = np.linalg.norm(center.c - d.c)
            if dist <= R:
                neibs[ind] = d
            ind += 1
        # print('neibs', len(neibs), 'are here')

        for _,v in neibs.items():
            x,y = toc_ds(v.c[0], v.c[1])
            # print(x,y)
            pygame.draw.circle(screen, (255,0,0), (x,y), 3)

        indexes = [k for k,v in neibs.items()]

        prev_center = center
        center_ind = random.randint(0, len(indexes)-1)
        center = neibs[indexes[center_ind]]
        cur_dist = 999999

        for d_ind, d in neibs.items():
            total_dist = 0
            for _, a in neibs.items():
                total_dist += np.linalg.norm(d.c - a.c)
            if total_dist < cur_dist:
                center = d
                center_ind = d_ind
                cur_dist = total_dist

        neibs_to_delete = dict(neibs)
        neibs = {center_ind:center}

        pygame.display.update()

    clusters[center] = neibs_to_delete
    # print(clusters)
    to_delete = [ind for ind, _ in neibs_to_delete.items()]
    # print(to_delete)
    dataset = np.delete(dataset, to_delete)
