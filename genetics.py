import matplotlib.pyplot as plt
from math import sin, sqrt
import random

MUTATION_RATE = 0.9

class EngFunc:
    @classmethod
    def value(self, x, y):
        sqrt1 = sqrt(abs(x/2 + (y+47)))
        sqrt2 = sqrt(abs(x - (y+47)))
        return -(y+47) * sin(sqrt1) - x * sin(sqrt2)

    @classmethod
    def err(self, x, y):
        return self.value(x,y) - self.value(512, 404.2319)

class O:
    def __init__(self, size):
        self.cur_err = 0

        self.genom = []

        for i in range(size):
            self.genom.append(random.randint(0,1))

    def value(self):
        return int("".join(str(x) for x in self.genom), 2)

    def value_as_xy(self):
        if len(self.genom) % 2 != 0:
            raise Exception
        res = []
        f_half = self.genom[int(len(self.genom)/2)-1::-1]
        s_half = self.genom[:int(len(self.genom)/2)-1:-1]

        res.append(int("".join(str(x) for x in f_half), 2))
        res.append(int("".join(str(x) for x in s_half), 2))
        return res

    def mutate(self):
        should_mutate = random.random()
        # print(should_mutate)
        if should_mutate <= MUTATION_RATE:
            gen_ind = random.randint(0, len(self.genom) - 1)
            self.genom[gen_ind] = abs(self.genom[gen_ind] - 1)

def classic_crossover(o1, o2):
    cut_point_ind = random.randint(0, len(o1.genom) - 1)
    new1_gene = o1.genom[cut_point_ind:] + o2.genom[:cut_point_ind]

    new2_gene = o2.genom[cut_point_ind:] + o1.genom[:cut_point_ind]

    new1 = O(len(o1.genom))
    new1.genom = new1_gene
    new2 = O(len(o1.genom))
    new2.genom = new2_gene

    res = [new1, new2]
    return res

def genitor_crossover(o1, o2):
    cut_point_ind = random.randint(0, len(o1.genom) - 1)

    new = O(len(o1.genom))

    o1_gene = o1.genom[:cut_point_ind]
    o2_gene = o2.genom[cut_point_ind:]

    new.genom = o1_gene + o2_gene
    return new

err_arr_classic = []
def classic():
    population_size = 100
    population = []

    upper_border = 512
    genom_size = bin(upper_border)
    genom_size = len(genom_size) - 3

    for i in range(population_size):
        population.append(O(genom_size * 2))

    print(EngFunc.value(512, 404.2319))

    steps = 10000
    counter = 0
    best_err = 99999
    okey_err = upper_border / 100
    while best_err > okey_err and counter < steps:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = EngFunc.err(x, y)
        population.sort(key=lambda x : x.cur_err)
        if counter % 100 == 0:
            err_arr_classic.append(population[0].cur_err)
            print('pop #' + str(counter), 'lowest err: {:.5}'.format(population[0].cur_err))

        mean_prisp = 0
        for el in population:
            mean_prisp += el.cur_err
        mean_prisp /= len(population)

        prop_tmp = []
        for el in population:
            el_prisp = el.cur_err/mean_prisp
            cel = int(el_prisp // mean_prisp)
            drob = el_prisp % mean_prisp
            for _ in range(cel):
                prop_tmp.append((population.index(el), el))
            r = random.random()
            if r < drob:
                prop_tmp.append((population.index(el), el))

        f_parent = random.randint(0, len(prop_tmp) - 1)
        s_parent = random.randint(0, len(prop_tmp) - 1)
        while f_parent == s_parent:
            s_parent = random.randint(0, len(prop_tmp) - 1)

        new1, new2 = classic_crossover(prop_tmp[f_parent][1], prop_tmp[s_parent][1])
        new1.mutate()
        new2.mutate()
        n1xy = new1.value_as_xy()
        new1.cur_err = EngFunc.err(n1xy[0], n1xy[1])
        n2xy = new2.value_as_xy()
        new2.cur_err = EngFunc.err(n2xy[0], n2xy[1])
        population[prop_tmp[f_parent][0]] = new1
        population[prop_tmp[s_parent][0]] = new2

        population.sort(key=lambda x : x.cur_err)

        best_err = population[0].cur_err
        counter += 1

    print('-------')
    print('pop #' + str(counter), 'err: {:.3},'.format(population[0].cur_err), '{:.3}%'.format((population[0].cur_err/upper_border)* 100))
    print('true best value', EngFunc.value(512, 404.2319))
    best_o_x, best_o_y = population[0].value_as_xy()

    print('gene best value', EngFunc.value(best_o_x, best_o_y))


err_arr_genitor = []
def genitor():
    population_size = 100
    population = []

    upper_border = 512
    genom_size = bin(upper_border)
    genom_size = len(genom_size) - 3

    for i in range(population_size):
        population.append(O(genom_size * 2))

    print(EngFunc.value(512, 404.2319))

    steps = 10000
    counter = 0
    best_err = 99999
    okey_err = upper_border / 100
    while best_err > okey_err and counter < steps:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = EngFunc.err(x, y)
        population.sort(key=lambda x : x.cur_err)
        if counter % 100 == 0:
            err_arr_genitor.append(population[0].cur_err)
            print('pop #' + str(counter), 'lowest err: {:.5}'.format(population[0].cur_err))

        f_parent_ind = random.randint(0, len(population) - 1)
        s_parent_ind = random.randint(0, len(population) - 1)

        new_o = genitor_crossover(population[f_parent_ind], population[s_parent_ind])
        new_o.mutate()
        population.sort(key=lambda x : x.cur_err)

        population[-1] = new_o

        best_err = population[0].cur_err
        counter += 1

    print('-------')
    print('pop #' + str(counter), 'err: {:.3},'.format(population[0].cur_err), '{:.3}%'.format((population[0].cur_err/upper_border)* 100))
    print('true best value', EngFunc.value(512, 404.2319))
    best_o_x, best_o_y = population[0].value_as_xy()

    print('gene best value', EngFunc.value(best_o_x, best_o_y))



classic()
genitor()
plt.plot(err_arr_classic, 'r-')
# plt.show()
plt.plot(err_arr_genitor, 'g-')
plt.show()