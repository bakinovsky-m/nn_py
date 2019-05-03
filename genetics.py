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

def crossover(o1, o2):
    cut_point_ind = random.randint(0, len(o1.genom) - 1)
    new1_gene = o1.genom[cut_point_ind:] + o2.genom[:cut_point_ind]

    new2_gene = o2.genom[cut_point_ind:] + o1.genom[:cut_point_ind]

    new1 = O(len(o1.genom))
    new1.genom = new1_gene
    new2 = O(len(o1.genom))
    new2.genom = new2_gene

    res = [new1, new2]
    return res

population_size = 100
population = []

upper_border = 512
genom_size = bin(upper_border)
genom_size = len(genom_size) - 3

for i in range(population_size):
    population.append(O(genom_size * 2))

print(EngFunc.value(512, 404.2319))

turnir_size = 2
steps = 100000
counter = 0
best_err = 99999
okey_err = upper_border / 100
while best_err > okey_err and counter < steps:
    for el in population:
        x, y = el.value_as_xy()
        el.cur_err = EngFunc.err(x, y)
    population.sort(key=lambda x : x.cur_err)
    if counter % 100 == 0:
        print('pop #' + str(counter), 'lowest err: {:.5}'.format(population[0].cur_err))

    turnir = []
    while len(turnir) <= len(population):
        f_o_ind = random.randint(0, len(population)-1)
        f_o = population[f_o_ind]
        s_o_ind = random.randint(0, len(population)-1)
        while f_o_ind == s_o_ind:
            s_o_ind = random.randint(0, len(population)-1)
        s_o = population[s_o_ind]
        if f_o.cur_err < s_o.cur_err:
            turnir.append((f_o_ind, f_o))
        else:
            turnir.append((s_o_ind, s_o))

    f_parent = random.randint(0, len(turnir) - 1)
    s_parent = random.randint(0, len(turnir) - 1)
    while f_parent == s_parent:
        s_parent = random.randint(0, len(turnir) - 1)

    new1, new2 = crossover(turnir[f_parent][1], turnir[s_parent][1])
    new1.mutate()
    new2.mutate()
    n1xy = new1.value_as_xy()
    new1.cur_err = EngFunc.err(n1xy[0], n1xy[1])
    n2xy = new2.value_as_xy()
    new2.cur_err = EngFunc.err(n2xy[0], n2xy[1])
    # population[turnir[f_parent][0]] = new1
    # population[turnir[s_parent][0]] = new2
    population.append(new1)
    population.append(new2)

    population.sort(key=lambda x : x.cur_err)

    population = population[:population_size]

    best_err = population[0].cur_err
    counter += 1

print('-------')
print('pop #' + str(counter), 'err: {:.3},'.format(population[0].cur_err), '{:.3}%'.format((population[0].cur_err/upper_border)* 100))
