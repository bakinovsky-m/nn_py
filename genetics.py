import matplotlib.pyplot as plt
from math import sin, sqrt
import random

MUTATION_RATE = 0.9
ACCURACY = 4

class EngFunc:
    def __init__(self):
        self.upper_border = 512
    def value(self, x, y):
        sqrt1 = sqrt(abs(x/2 + (y+47)))
        sqrt2 = sqrt(abs(x - (y+47)))
        return -(y+47) * sin(sqrt1) - x * sin(sqrt2)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(512, 404.2319))

class SphereFunc:
    def __init__(self):
        self.upper_border = 4
    N = 10
    def value(self, x, y):
        res = 0
        res += x ** 2
        res += y ** 2
        return float(res)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(0,0))

class ShapherN2Func:
    def __init__(self):
        self.upper_border = 100
    def value(self, x, y):
        up = sin(x**2 + y**2) ** 2 - 0.5
        bottom = (1 + 0.001 * (x**2 + y**2)) ** 2
        return 0.5 * up/bottom

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(0,0))

class RosenbrockFunc:
    def __init__(self):
        self.upper_border = 512
    def value(self, x, y):
        return float((1-x)**2 + 100*(y-x**2)**2)

    def err(self, x, y):
        return abs(self.value(x,y) - self.value(1,1))

class O:
    def __init__(self, size):
        self.cur_err = 0

        self.genom = []

        for i in range(size):
            self.genom.append(random.randint(0,1))

        self.acc = ACCURACY

    def value(self):
        return int("".join(str(x) for x in self.genom), 2)

    def value_as_xy(self):
        if len(self.genom) % 2 != 0:
            raise Exception
        res = []
        # f_half = self.genom[int(len(self.genom)/2)-1::-1]
        # s_half = self.genom[:int(len(self.genom)/2)-1:-1]

        genom_len = len(self.genom)

        f_half = self.genom[:int(genom_len/2)-self.acc]
        s_half = self.genom[int(genom_len/2):-self.acc]

        f_float = self.genom[int(genom_len/2)-self.acc:int(genom_len/2)]
        f_f = 0
        c = 0
        for f in f_float:
            f_f += f * (2**c)
            c+=1
        f_f *= 10 ** c
        s_float = self.genom[-self.acc:]
        s_f = 0
        c = 0
        for f in s_float:
            s_f += f * (2**c)
            c+=1
        s_f *= 10 ** c

        res.append(int("".join(str(x) for x in f_half), 2) + f_f)
        res.append(int("".join(str(x) for x in s_half), 2) + s_f)
        return res

    def mutate(self):
        should_mutate = random.random()
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

    res = (new1, new2)
    return res

def genitor_crossover(o1, o2):
    cut_point_ind1 = random.randint(0, len(o1.genom) - 1)
    cut_point_ind2 = random.randint(0, len(o1.genom) - 1)

    new = O(len(o1.genom))

    if cut_point_ind2 < cut_point_ind1:
        cut_point_ind1, cut_point_ind2 = cut_point_ind2, cut_point_ind1

    o1_gene = o1.genom[:cut_point_ind1]
    o2_gene = o2.genom[cut_point_ind1:cut_point_ind2]
    o3_gene = o1.genom[cut_point_ind2:]
    

    new.genom = o1_gene + o2_gene + o3_gene
    return new

# FUNC = EngFunc()
FUNC = SphereFunc()
# FUNC = ShapherN2Func()
# FUNC = RosenbrockFunc()

population_size = 100
# upper_border = 16
upper_border = FUNC.upper_border
genom_size = bin(upper_border)
# print(genom_size)
genom_size = len(genom_size) - 2 + 2*ACCURACY
# print(genom_size)
# sys.exit(0)
steps = 10000
okey_err = upper_border / 1000

err_arr_classic = []
def classic():
    population = []
    for i in range(population_size):
        population.append(O(genom_size * 2))

    counter = 0
    best_err = 99999
    while best_err > okey_err and counter < steps:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = FUNC.err(x, y)
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
        if len(prop_tmp) < 2:
            prop_tmp.append((0, population[0]))
            prop_tmp.append((1, population[1]))

        f_parent = random.randint(0, len(prop_tmp) - 1)
        s_parent = random.randint(0, len(prop_tmp) - 1)
        while f_parent == s_parent:
            s_parent = random.randint(0, len(prop_tmp) - 1)

        new1, new2 = classic_crossover(prop_tmp[f_parent][1], prop_tmp[s_parent][1])
        new1.mutate()
        new2.mutate()
        n1xy = new1.value_as_xy()
        new1.cur_err = FUNC.err(n1xy[0], n1xy[1])
        n2xy = new2.value_as_xy()
        new2.cur_err = FUNC.err(n2xy[0], n2xy[1])
        population[prop_tmp[f_parent][0]] = new1
        population[prop_tmp[s_parent][0]] = new2

        population.sort(key=lambda x : x.cur_err)

        best_err = population[0].cur_err
        counter += 1

    print('-------')
    print('pop #' + str(counter), 'err: {:.3},'.format(population[0].cur_err), '{:.3}%'.format((population[0].cur_err/upper_border)* 100))
    print('true best value', FUNC.value(512, 404.2319))
    best_o_x, best_o_y = population[0].value_as_xy()

    print('gene best value', FUNC.value(best_o_x, best_o_y))


err_arr_genitor = []
def genitor():
    population = []

    for i in range(population_size):
        population.append(O(genom_size * 2))

    counter = 0
    best_err = 99999
    while best_err > okey_err and counter < steps:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = FUNC.err(x, y)
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
    print('true best value', FUNC.value(512, 404.2319))
    best_o_x, best_o_y = population[0].value_as_xy()

    print('gene best value', FUNC.value(best_o_x, best_o_y))



classic()
genitor()
plt.plot(err_arr_classic, 'r-')
# plt.show()
plt.plot(err_arr_genitor, 'g-')
plt.show()